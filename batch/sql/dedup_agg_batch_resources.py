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


async def process_chunk_agg_batch_resources(counter, db, start, end, quiet=True):
    if start is None:
        assert end is not None
        end_batch_id, end_resource_id = end

        where_statement = '''
WHERE (aggregated_batch_resources_v2.batch_id < %s) OR
      (aggregated_batch_resources_v2.batch_id = %s AND aggregated_batch_resources_v2.resource_id < %s)
'''
        query_args = [end_batch_id,
                      end_batch_id, end_resource_id,
                      ]
    elif end is None:
        assert start is not None
        start_batch_id, start_resource_id = start

        where_statement = '''
WHERE (aggregated_batch_resources_v2.batch_id > %s) OR
      (aggregated_batch_resources_v2.batch_id = %s AND aggregated_batch_resources_v2.resource_id >= %s)
'''
        query_args = [start_batch_id,
                      start_batch_id, start_resource_id,
                      ]
    else:
        assert start is not None and end is not None
        start_batch_id, start_resource_id = start
        end_batch_id, end_resource_id = end

        where_statement = '''
WHERE ((aggregated_batch_resources_v2.batch_id > %s) OR
       (aggregated_batch_resources_v2.batch_id = %s AND aggregated_batch_resources_v2.resource_id >= %s))
  AND ((aggregated_batch_resources_v2.batch_id < %s) OR
       (aggregated_batch_resources_v2.batch_id = %s AND aggregated_batch_resources_v2.resource_id < %s))
'''
        query_args = [start_batch_id,
                      start_batch_id, start_resource_id,
                      end_batch_id,
                      end_batch_id, end_resource_id,
                      ]

    query = f'''
UPDATE aggregated_batch_resources_v2
SET migrated = 1
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def audit_changes(db):
    batch_audit_start = time.time()
    print('starting auditing batch records')

    chunk_offsets = [None]
    for offset in await find_chunk_offsets_for_audit(db, 100):
        chunk_offsets.append(offset)
    chunk_offsets = list(zip(chunk_offsets[:-1], chunk_offsets[1:]))

    async def _process_audit_chunk(db, start_id, end_id):
        if start_id is not None and end_id is not None:
            where_statement = '''
WHERE batch_id >= %s AND batch_id < %s
        '''
            where_args = [start_id, end_id]
        elif start_id is None and end_id is not None:
            where_statement = 'WHERE batch_id < %s'
            where_args = [end_id]
        else:
            assert start_id is not None and end_id is None
            where_statement = 'WHERE batch_id >= %s'
            where_args = [start_id]

        bad_batch_records = db.select_and_fetchall(
            f'''
SELECT old.batch_id, old.deduped_resource_id, old.`usage`, new.`usage`, ABS(new.`usage` - old.`usage`) AS usage_diff
FROM (
  SELECT batch_id, deduped_resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_batch_resources_v2
  LEFT JOIN resources ON resources.resource_id = aggregated_batch_resources_v2.resource_id
  {where_statement}
  GROUP BY batch_id, deduped_resource_id
  LOCK IN SHARE MODE
) AS old
LEFT JOIN (
  SELECT batch_id, deduped_resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_batch_resources_v3
  LEFT JOIN resources ON resources.resource_id = aggregated_batch_resources_v3.resource_id
  {where_statement}
  GROUP BY batch_id, resource_id
  LOCK IN SHARE MODE
) AS new ON old.batch_id = new.batch_id AND old.deduped_resource_id = new.deduped_resource_id
WHERE new.`usage` != old.`usage`
LIMIT 100;
''',
            where_args + where_args)

        bad_batch_records = [record async for record in bad_batch_records]
        failing_batches = []
        for record in bad_batch_records:
            print(f'found bad batch record {record}')
            failing_batches.append((record['batch_id']))

        if bad_batch_records:
            raise Exception(f'errors found in audit')

    if chunk_offsets != [(None, None)]:
        random.shuffle(chunk_offsets)

        await bounded_gather(
            *[functools.partial(_process_audit_chunk, db, start_offset, end_offset)
              for start_offset, end_offset in chunk_offsets],
            parallelism=10
        )

    print(f'finished auditing batch records in {time.time() - batch_audit_start}s')


async def find_chunk_offsets(db, size):
    @transaction(db)
    async def _find_chunks(tx) -> List[Optional[Tuple[int, int, str]]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        query = f'''
SELECT t.batch_id, t.resource_id FROM (
  SELECT batch_id, resource_id
  FROM aggregated_batch_resources_v2
  ORDER BY batch_id ASC, resource_id ASC
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [(offset['batch_id'], offset['resource_id']) async for offset in offsets]
        offsets.append(None)

        print(f'found chunk offsets in {round(time.time() - start_time, 4)}s')
        return offsets

    return await _find_chunks()


async def find_chunk_offsets_for_audit(db, size):
    @transaction(db)
    async def _find_chunks(tx) -> List[Optional[Tuple[int, int, str]]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        query = f'''
SELECT t.batch_id FROM (
  SELECT batch_id
  FROM aggregated_batch_resources_v2
  ORDER BY batch_id ASC
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [(offset['batch_id']) async for offset in offsets]
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
            await process_chunk_agg_batch_resources(chunk_counter, db, start_offset, end_offset, quiet=False)

        print(f'finished burn-in in {time.time() - burn_in_start}s')

        parallel_insert_start = time.time()

        # 4 core database, parallelism = 10 maxes out CPU
        await bounded_gather(
            *[functools.partial(process_chunk_agg_batch_resources, chunk_counter, db, start_offset, end_offset, quiet=False)
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
