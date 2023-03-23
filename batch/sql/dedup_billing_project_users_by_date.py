import asyncio
import functools
import os
import random
import time
from typing import List, Optional, Tuple

from gear import Database, transaction
from hailtop.utils import bounded_gather


MYSQL_CONFIG_FILE = os.environ.get('MYSQL_CONFIG_FILE')


class Counter:
    def __init__(self):
        self.n = 0


async def process_chunk(counter, db, query, query_args, start, end, quiet=True):
    start_time = time.time()

    await db.just_execute(query, query_args)

    if not quiet and counter.n % 100 == 0:
        print(f'processed chunk ({start}, {end}) in {time.time() - start_time}s')

    counter.n += 1
    if counter.n % 500 == 0:
        print(f'processed {counter.n} complete chunks')


async def process_chunk_agg_bp_users_by_date(counter, db, start, end, quiet=True):
    if start is None:
        assert end is not None
        end_billing_date, end_billing_project, end_user, end_resource_id = end

        where_statement = '''
WHERE (aggregated_billing_project_user_resources_by_date_v2.billing_date < %s) OR
      (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project < %s) OR
      (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` < %s) OR
      (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` = %s AND aggregated_billing_project_user_resources_by_date_v2.resource_id < %s)
'''
        query_args = [end_billing_date,
                      end_billing_date, end_billing_project,
                      end_billing_date, end_billing_project, end_user,
                      end_billing_date, end_billing_project, end_user, end_resource_id,
                      ]
    elif end is None:
        assert start is not None
        start_billing_date, start_billing_project, start_user, start_resource_id = start

        where_statement = '''
WHERE (aggregated_billing_project_user_resources_by_date_v2.billing_date > %s) OR
      (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project > %s) OR
      (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` > %s) OR
      (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` = %s AND aggregated_billing_project_user_resources_by_date_v2.resource_id >= %s)
'''
        query_args = [start_billing_date,
                      start_billing_date, start_billing_project,
                      start_billing_date, start_billing_project, start_user,
                      start_billing_date, start_billing_project, start_user, start_resource_id,
                      ]
    else:
        assert start is not None and end is not None
        start_billing_date, start_billing_project, start_user, start_resource_id = start
        end_billing_date, end_billing_project, end_user, end_resource_id = end

        where_statement = '''
WHERE ((aggregated_billing_project_user_resources_by_date_v2.billing_date > %s) OR
       (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project > %s) OR
       (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` > %s) OR
       (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` = %s AND aggregated_billing_project_user_resources_by_date_v2.resource_id >= %s))
  AND ((aggregated_billing_project_user_resources_by_date_v2.billing_date < %s) OR
       (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project < %s) OR
       (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` < %s) OR
       (aggregated_billing_project_user_resources_by_date_v2.billing_date = %s AND aggregated_billing_project_user_resources_by_date_v2.billing_project = %s AND aggregated_billing_project_user_resources_by_date_v2.`user` = %s AND aggregated_billing_project_user_resources_by_date_v2.resource_id < %s))
'''
        query_args = [start_billing_date,
                      start_billing_date, start_billing_project,
                      start_billing_date, start_billing_project, start_user,
                      start_billing_date, start_billing_project, start_user, start_resource_id,
                      end_billing_date,
                      end_billing_date, end_billing_project,
                      end_billing_date, end_billing_project, end_user,
                      end_billing_date, end_billing_project, end_user, end_resource_id,
                      ]

    query = f'''
UPDATE aggregated_billing_project_user_resources_by_date_v2
SET migrated = 1
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def audit_changes(db):
    bp_user_by_date_audit_start = time.time()
    print('starting auditing bp user by date records')

    bad_bp_user_by_date_records = db.select_and_fetchall(
        '''
SELECT old.billing_date, old.billing_project, old.`user`, old.`usage`, new.`usage`, new.`usage` - old.`usage` as usage_diff
FROM (
  SELECT billing_date, billing_project, `user`, deduped_resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
  FROM aggregated_billing_project_user_resources_by_date_v2
  LEFT JOIN resources ON aggregated_billing_project_user_resources_by_date_v2.resource_id = resources.resource_id
  GROUP BY billing_date, billing_project, `user`, deduped_resource_id
) AS old
LEFT JOIN (
  SELECT billing_date, billing_project, `user`, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
  FROM aggregated_billing_project_user_resources_by_date_v3
  GROUP BY billing_date, billing_project, `user`, resource_id
) AS new ON old.billing_date = new.billing_date and old.billing_project = new.billing_project AND old.`user` = new.`user` and old.deduped_resource_id = new.resource_id
WHERE new.`usage` != old.`usage`
LIMIT 100;
''')

    bad_bp_user_by_date_records = [record async for record in bad_bp_user_by_date_records]
    failing_bp_users_by_date = []
    for record in bad_bp_user_by_date_records:
        print(f'found bad bp user by date record {record}')
        failing_bp_users_by_date.append((record['billing_date'], record['billing_project'], record['user']))

    print(f'finished auditing bp user by date records in {time.time() - bp_user_by_date_audit_start}s')

    if failing_bp_users_by_date:
        raise Exception("audit failed")


async def find_chunk_offsets(db, size):
    @transaction(db)
    async def _find_chunks(tx) -> List[Optional[Tuple[int, int, str]]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        query = f'''
SELECT t.billing_date, t.billing_project, t.`user`, t.resource_id FROM (
  SELECT billing_date, billing_project, `user`, resource_id
  FROM aggregated_billing_project_user_resources_by_date_v2
  ORDER BY billing_date ASC, billing_project ASC, `user` ASC, resource_id ASC
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [(offset['billing_date'], offset['billing_project'], offset['user'], offset['resource_id']) async for offset in offsets]
        offsets.append(None)

        print(f'found chunk offsets in {round(time.time() - start_time, 4)}s')
        return offsets

    return await _find_chunks()


async def run_migration(db, chunk_size):
    populate_start_time = time.time()

    chunk_counter = Counter()
    chunk_offsets = [None]
    for offset in await find_chunk_offsets(db, chunk_size):
        chunk_offsets.append(offset)

    chunk_offsets = list(zip(chunk_offsets[:-1], chunk_offsets[1:]))

    if chunk_offsets != [(None, None)]:
        print(f'found {len(chunk_offsets)} chunks to process')

        random.shuffle(chunk_offsets)

        burn_in_start = time.time()
        n_burn_in_chunks = 10000

        burn_in_chunk_offsets = chunk_offsets[:n_burn_in_chunks]
        chunk_offsets = chunk_offsets[n_burn_in_chunks:]

        for start_offset, end_offset in burn_in_chunk_offsets:
            await process_chunk_agg_bp_users_by_date(chunk_counter, db, start_offset, end_offset, quiet=False)

        print(f'finished burn-in in {time.time() - burn_in_start}s')

        parallel_insert_start = time.time()

        # 4 core database, parallelism = 10 maxes out CPU
        await bounded_gather(
            *[functools.partial(process_chunk_agg_bp_users_by_date, chunk_counter, db, start_offset, end_offset, quiet=False)
              for start_offset, end_offset in chunk_offsets],
            parallelism=10
        )
        print(
            f'took {time.time() - parallel_insert_start}s to insert the remaining complete records in parallel ({(chunk_size * len(chunk_offsets)) / (time.time() - parallel_insert_start)}) attempts / sec')

    print(f'finished populating records in {time.time() - populate_start_time}s')


async def main(chunk_size=100):
    db = Database()
    await db.async_init(config_file=MYSQL_CONFIG_FILE)

    start_time = time.time()

    try:
        migration_start_time = time.time()

        await run_migration(db, chunk_size)

        print(f'finished populating records in {time.time() - migration_start_time}s')

        audit_start_time = time.time()
        await audit_changes(db)
        print(f'finished auditing changes in {time.time() - audit_start_time}')
    finally:
        print(f'finished migration in {time.time() - start_time}s')
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
