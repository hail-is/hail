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


def offsets_to_where_statement(table, primary_key, start_offset, end_offset):
    assert start_offset != end_offset, str((start_offset, end_offset))

    if start_offset is None:
        start_offset = [None] * len(primary_key)
    if end_offset is None:
        end_offset = [None] * len(primary_key)

    def build_statement(offset, is_start):
        where_statement = []
        query_args = []
        for i, key in enumerate(primary_key):
            conds = [f'{table}.{primary_key[prev_i]} = %s' for prev_i in range(i)]
            query_args += offset[:i]
            if is_start:
                if i == len(primary_key) - 1:
                    conds.append(f'{table}.{primary_key[i]} >= %s')
                    query_args.append(offset[i])
                else:
                    conds.append(f'{table}.{primary_key[i]} > %s')
                    query_args.append(offset[i])
            else:
                conds.append(f'{table}.{primary_key[i]} < %s')
                query_args.append(offset[i])
            cond = '(' + ' AND '.join(conds) + ')'
            where_statement.append(cond)
        return where_statement, query_args

    if None in start_offset:
        assert end_offset[0] is not None
        where_statements, query_args = build_statement(end_offset, False)
        where_statement = f'WHERE {" OR ".join(where_statements)}'
    elif None in end_offset:
        assert start_offset[0] is not None
        where_statements, query_args = build_statement(start_offset, True)
        where_statement = f'WHERE {" OR ".join(where_statements)}'
    else:
        start_where_statements, start_query_args = build_statement(start_offset, True)
        end_where_statements, end_query_args = build_statement(end_offset, False)
        where_statement = f'WHERE ({" OR ".join(start_where_statements)}) AND ({" OR ".join(end_where_statements)})'
        query_args = start_query_args + end_query_args

    return where_statement, query_args


async def process_chunk(counter, db, query, query_args, start, end, quiet=True):
    start_time = time.time()

    await db.just_execute(query, query_args)

    if not quiet and counter.n % 100 == 0:
        print(f'processed chunk ({start}, {end}) in {time.time() - start_time}s')

    counter.n += 1
    if counter.n % 500 == 0:
        print(f'processed {counter.n} complete chunks')


async def process_chunk_attempt_resources(counter, db, start, end, quiet=True):
    where_statement, query_args = offsets_to_where_statement('attempt_resources',
                                                             ['batch_id', 'job_id', 'attempt_id', 'resource_id'],
                                                             start,
                                                             end)
    query = f'''
UPDATE attempt_resources
LEFT JOIN resources ON attempt_resources.resource_id = resources.resource_id
SET attempt_resources.deduped_resource_id = resources.deduped_resource_id
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def process_chunk_aggregated_billing_project_user_resources(counter, db, start, end, quiet=True):
    where_statement, query_args = offsets_to_where_statement('aggregated_billing_project_user_resources_v2',
                                                             ['billing_project', 'user', 'resource_id', 'token'],
                                                             start,
                                                             end)
    query = f'''
UPDATE aggregated_billing_project_user_resources_v2
SET migrated = 1
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def process_chunk_aggregated_billing_project_user_resources_by_date(counter, db, start, end, quiet=True):
    where_statement, query_args = offsets_to_where_statement('aggregated_billing_project_user_resources_by_date_v2',
                                                             ['billing_date', 'billing_project', 'user', 'resource_id', 'token'],
                                                             start,
                                                             end)
    query = f'''
UPDATE aggregated_billing_project_user_resources_by_date_v2
SET migrated = 1
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def process_chunk_aggregated_batch_resources(counter, db, start, end, quiet=True):
    where_statement, query_args = offsets_to_where_statement('aggregated_batch_resources_v2',
                                                             ['batch_id', 'resource_id', 'token'],
                                                             start,
                                                             end)
    query = f'''
UPDATE aggregated_batch_resources_v2
SET migrated = 1
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def process_chunk_aggregated_job_resources(counter, db, start, end, quiet=True):
    where_statement, query_args = offsets_to_where_statement('aggregated_job_resources_v2',
                                                             ['batch_id', 'job_id' ,'resource_id'],
                                                             start,
                                                             end)
    query = f'''
UPDATE aggregated_job_resources_v2
SET migrated = 1
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def audit_changes(db):
    job_audit_start = time.time()
    print('starting auditing job records')

    bad_job_records = db.select_and_fetchall(
        '''
SELECT old.batch_id, old.job_id, old.cost, new.cost, ABS(new.cost - old.cost) AS cost_diff
FROM (
  SELECT batch_id, job_id, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT batch_id, job_id, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_job_resources_v2
    GROUP BY batch_id, job_id, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY batch_id, job_id
) AS old
LEFT JOIN (
  SELECT batch_id, job_id, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT batch_id, job_id, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_job_resources_v3
    GROUP BY batch_id, job_id, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY batch_id, job_id
) AS new ON old.batch_id = new.batch_id AND old.job_id = new.job_id
WHERE ABS(new.cost - old.cost) = 0
LIMIT 100;
''')

    bad_job_records = [record async for record in bad_job_records]
    failing_job_ids = []
    for record in bad_job_records:
        print(f'found bad job record {record}')
        failing_job_ids.append((record['batch_id'], record['job_id']))

    print(f'finished auditing job records in {time.time() - job_audit_start}s')

    batch_audit_start = time.time()
    print('starting auditing batch records')

    bad_batch_records = db.select_and_fetchall(
        '''
SELECT old.batch_id, old.cost, new.cost, ABS(new.cost - old.cost) AS cost_diff
FROM (
  SELECT batch_id, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT batch_id, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_batch_resources_v2
    GROUP BY batch_id, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY batch_id
) AS old
LEFT JOIN (
  SELECT batch_id, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT batch_id, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_batch_resources_v3
    GROUP BY batch_id, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY batch_id
) AS new ON old.batch_id = new.batch_id
WHERE ABS(new.cost - old.cost) = 0
LIMIT 100;
''')

    bad_batch_records = [record async for record in bad_batch_records]
    failing_batch_ids = []
    for record in bad_batch_records:
        print(f'found bad batch record {record}')
        failing_batch_ids.append(record['batch_id'])

    print(f'finished auditing batch records in {time.time() - batch_audit_start}s')

    bp_user_audit_start = time.time()
    print('starting auditing billing project user records')

    bad_bp_user_records = db.select_and_fetchall(
        '''
SELECT old.billing_project, old.user, old.cost, new.cost, ABS(new.cost - old.cost) AS cost_diff
FROM (
  SELECT billing_project, user, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT billing_project, user, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_billing_project_user_resources_by_date_v2
    GROUP BY billing_project, user, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY billing_project, user
) AS old
LEFT JOIN (
  SELECT billing_project, user, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT billing_project, user, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_billing_project_user_resources_by_date_v3
    GROUP BY billing_project, user, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY billing_project, user
) AS new ON old.billing_project = new.billing_project AND old.user = new.user
WHERE ABS(new.cost - old.cost) = 0
LIMIT 100;
''')

    bad_bp_user_records = [record async for record in bad_bp_user_records]
    failing_bp_users = []
    for record in bad_bp_user_records:
        print(f'found bad bp user record {record}')
        failing_bp_users.append((record['billing_project'], record['user']))

    print(f'finished auditing bp user records in {time.time() - bp_user_audit_start}s')

    bp_user_by_date_audit_start = time.time()
    print('starting auditing bp user by date records')

    bad_bp_user_by_date_records = db.select_and_fetchall(
        '''
SELECT old.billing_project, old.user, old.cost, new.cost, ABS(new.cost - old.cost) AS cost_diff
FROM (
  SELECT billing_date, billing_project, user, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT billing_date, billing_project, user, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_billing_project_user_resources_by_date_v2
    GROUP BY billing_date, billing_project, user, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY billing_date, billing_project, user
) AS old
LEFT JOIN (
  SELECT billing_date, billing_project, user, CAST(COALESCE(SUM(`usage` * rate), 0) AS SIGNED) AS cost
  FROM (
    SELECT billing_date, billing_project, user, resource_id, COALESCE(SUM(`usage`), 0) AS `usage`
    FROM aggregated_billing_project_user_resources_by_date_v3
    GROUP BY billing_date, billing_project, user, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY billing_date, billing_project, user
) AS new ON old.billing_project = new.billing_project AND old.user = new.user
WHERE ABS(new.cost - old.cost) = 0
LIMIT 100;
''')

    bad_bp_user_by_date_records = [record async for record in bad_bp_user_by_date_records]
    failing_bp_users_by_date = []
    for record in bad_bp_user_by_date_records:
        print(f'found bad bp user by date record {record}')
        failing_bp_users_by_date.append((record['billing_date'], record['billing_project'], record['user']))

    print(f'finished auditing bp user by date records in {time.time() - bp_user_by_date_audit_start}s')

    if failing_job_ids or failing_batch_ids or failing_bp_users or failing_bp_users_by_date:
        raise Exception(f'errors found in audit')


async def find_chunk_offsets(db, size, table, primary_key):
    @transaction(db)
    async def _find_chunks(tx) -> List[Optional[Tuple[int, int, str]]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        primary_key_str = ', '.join(primary_key)
        primary_key_t_str = ', '.join(f't.{key}' for key in primary_key)

        query = f'''
SELECT {primary_key_t_str} FROM (
  SELECT {primary_key_str}
  FROM {table}
  ORDER BY {primary_key_str}
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [tuple(offset[pk] for pk in primary_key) async for offset in offsets]
        offsets.append(None)

        print(f'found chunk offsets in {round(time.time() - start_time, 4)}s')
        return offsets

    return await _find_chunks()


async def run_migration(db, chunk_size, table, primary_key, processor_f):
    populate_start_time = time.time()

    chunk_counter = Counter()
    chunk_offsets = [None]
    for offset in await find_chunk_offsets(db, chunk_size, table, primary_key):
        chunk_offsets.append(offset)

    chunk_offsets = list(zip(chunk_offsets[:-1], chunk_offsets[1:]))

    if chunk_offsets != [(None, None)]:
        print(f'found {len(chunk_offsets)} chunks to process for table {table} with primary key {primary_key}')

        random.shuffle(chunk_offsets)

        burn_in_start = time.time()
        n_burn_in_chunks = 10000

        burn_in_chunk_offsets = chunk_offsets[:n_burn_in_chunks]
        chunk_offsets = chunk_offsets[n_burn_in_chunks:]

        for start_offset, end_offset in burn_in_chunk_offsets:
            await processor_f(chunk_counter, db, start_offset, end_offset, quiet=False)

        print(f'finished burn-in in {time.time() - burn_in_start}s for table {table} with primary key {primary_key}')

        parallel_insert_start = time.time()

        # 4 core database, parallelism = 10 maxes out CPU
        await bounded_gather(
            *[functools.partial(processor_f, chunk_counter, db, start_offset, end_offset, quiet=False)
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

        await run_migration(db,
                            chunk_size,
                            'attempt_resources',
                            ['batch_id', 'job_id', 'attempt_id', 'resource_id'],
                            process_chunk_attempt_resources)

        await run_migration(db,
                            chunk_size,
                            'aggregated_billing_project_user_resources_v2',
                            ['billing_project', 'user', 'resource_id', 'token'],
                            process_chunk_aggregated_billing_project_user_resources)

        await run_migration(db,
                            chunk_size,
                            'aggregated_billing_project_user_resources_by_date_v2',
                            ['billing_date', 'billing_project', 'user', 'resource_id', 'token'],
                            process_chunk_aggregated_billing_project_user_resources_by_date)

        await run_migration(db,
                            chunk_size,
                            'aggregated_batch_resources_v2',
                            ['batch_id', 'resource_id', 'token'],
                            process_chunk_aggregated_batch_resources)

        await run_migration(db,
                            chunk_size,
                            'aggregated_job_resources_v2',
                            ['batch_id', 'job_id', 'resource_id'],
                            process_chunk_aggregated_job_resources)

        print(f'finished populating records in {time.time() - migration_start_time}s')

        audit_start_time = time.time()
        await audit_changes(db)
        print(f'finished auditing changes in {time.time() - audit_start_time}')
    finally:
        print(f'finished migration in {time.time() - start_time}s')
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
