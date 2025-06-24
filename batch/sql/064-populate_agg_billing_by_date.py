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


def offsets_to_where_statement(start_offset, end_offset):
    assert start_offset != end_offset, str((start_offset, end_offset))

    if start_offset is None:
        start_batch_id, start_job_id, start_attempt_id = None, None, None
    else:
        start_batch_id, start_job_id, start_attempt_id = start_offset

    if end_offset is None:
        end_batch_id, end_job_id, end_attempt_id = None, None, None
    else:
        end_batch_id, end_job_id, end_attempt_id = end_offset

    if start_batch_id is None or start_job_id is None or start_attempt_id is None:
        assert end_batch_id
        where_cond = 'WHERE attempts.batch_id < %s OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id < %s) OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id < %s)'
        query_args = (end_batch_id, end_batch_id, end_job_id, end_batch_id, end_job_id, end_attempt_id)
    elif end_batch_id is None or end_job_id is None or end_attempt_id is None:
        assert start_batch_id
        where_cond = 'WHERE attempts.batch_id > %s OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id > %s) OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id >= %s)'
        query_args = (start_batch_id, start_batch_id, start_job_id, start_batch_id, start_job_id, start_attempt_id)
    else:
        where_cond = 'WHERE (attempts.batch_id > %s OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id > %s) OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id >= %s)) ' \
                     'AND (attempts.batch_id < %s OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id < %s) OR ' \
                     '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id < %s))'
        query_args = (start_batch_id, start_batch_id, start_job_id, start_batch_id, start_job_id, start_attempt_id,
                      end_batch_id, end_batch_id, end_job_id, end_batch_id, end_job_id, end_attempt_id)

    return (where_cond, query_args)


async def process_chunk(counter, db, start_offset, end_offset, quiet=True):
    start_time = time.time()

    where_cond, query_args = offsets_to_where_statement(start_offset, end_offset)

    await db.just_execute(
        f'''
UPDATE attempts
SET migrated = TRUE
{where_cond}
''',
        query_args)

    if not quiet and counter.n % 100 == 0:
        print(f'processed chunk ({start_offset}, {end_offset}) in {time.time() - start_time}s')

    counter.n += 1
    if counter.n % 500 == 0:
        print(f'processed {counter.n} complete chunks')


async def audit_changes(db):
    job_audit_start = time.time()
    print('starting auditing job records')

    bad_job_records = db.select_and_fetchall(
        '''
SELECT old.batch_id, old.job_id, old.cost, new.cost, ABS(new.cost - old.cost) AS cost_diff
FROM (
  SELECT batch_id, job_id, COALESCE(SUM(`usage` * rate), 0) AS cost
  FROM aggregated_job_resources
  LEFT JOIN batches ON batches.id = aggregated_job_resources.batch_id
  LEFT JOIN resources ON aggregated_job_resources.resource = resources.resource
  WHERE format_version >= 3
  GROUP BY batch_id, job_id
) AS old
LEFT JOIN (
  SELECT batch_id, job_id, COALESCE(SUM(`usage` * rate), 0) AS cost
  FROM aggregated_job_resources_v2
  LEFT JOIN resources ON aggregated_job_resources_v2.resource_id = resources.resource_id
  GROUP BY batch_id, job_id
) AS new ON old.batch_id = new.batch_id AND old.job_id = new.job_id
WHERE ABS(new.cost - old.cost) >= 0.00001
LIMIT 100;
''')

    bad_job_records = [record async for record in bad_job_records]
    failing_job_ids = []
    for record in bad_job_records:
        print(f'found bad job record {record}')

        # we had a bug in billing for job private instances that failed to activate
        # this was fixed in #10069
        maybe_bad_record = await db.select_and_fetchone(
            '''
SELECT * FROM attempts
WHERE batch_id = %s AND job_id = %s AND reason = "activation_timeout"
LIMIT 10;
''',
            (record['batch_id'], record['job_id']))
        if not maybe_bad_record:
            failing_job_ids.append((record['batch_id'], record['job_id']))

    print(f'finished auditing job records in {time.time() - job_audit_start}s')

    batch_audit_start = time.time()
    print('starting auditing batch records')

    bad_batch_records = db.select_and_fetchall(
        '''
SELECT old.batch_id, old.cost, new.cost, ABS(new.cost - old.cost) AS cost_diff
FROM (
  SELECT batch_id, COALESCE(SUM(`usage` * rate), 0) AS cost
  FROM aggregated_batch_resources
  LEFT JOIN batches ON batches.id = aggregated_batch_resources.batch_id
  LEFT JOIN resources ON aggregated_batch_resources.resource = resources.resource
  WHERE format_version >= 3
  GROUP BY batch_id
) AS old
LEFT JOIN (
  SELECT batch_id, COALESCE(SUM(`usage` * rate), 0) AS cost
  FROM aggregated_batch_resources_v2
  LEFT JOIN resources ON aggregated_batch_resources_v2.resource_id = resources.resource_id
  GROUP BY batch_id
) AS new ON old.batch_id = new.batch_id
WHERE ABS(new.cost - old.cost) >= 0.00001
LIMIT 100;
''')

    bad_batch_records = [record async for record in bad_batch_records]
    failing_batch_ids = []
    for record in bad_batch_records:
        print(f'found bad batch record {record}')

        # we had a bug in billing for job private instances that failed to activate
        # this was fixed in #10069
        maybe_bad_record = await db.select_and_fetchone(
            '''
SELECT * FROM attempts
WHERE batch_id = %s AND reason = "activation_timeout"
LIMIT 10;
''',
            (record['batch_id'],))
        if not maybe_bad_record:
            failing_batch_ids.append(record['batch_id'])

    print(f'finished auditing batch records in {time.time() - batch_audit_start}s')

    # cannot audit billing project records because they are partially filled in from batches with format version < 3

    if failing_job_ids or failing_batch_ids:
        raise Exception(f'errors found in audit')


async def find_chunk_offsets(db, size):
    @transaction(db)
    async def _find_chunks(tx) -> List[Optional[Tuple[int, int, str]]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        query = f'''
SELECT t.batch_id, t.job_id, t.attempt_id FROM (
  SELECT batch_id, job_id, attempt_id
  FROM attempts
  ORDER BY batch_id, job_id, attempt_id
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [(offset['batch_id'], offset['job_id'], offset['attempt_id']) async for offset in offsets]
        offsets.append(None)

        print(f'found chunk offsets in {round(time.time() - start_time, 4)}s')
        return offsets

    return await _find_chunks()


async def main(chunk_size=100):
    db = Database()
    await db.async_init(config_file=MYSQL_CONFIG_FILE)

    start_time = time.time()

    try:
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
                await process_chunk(chunk_counter, db, start_offset, end_offset, quiet=False)

            print(f'finished burn-in in {time.time() - burn_in_start}s')

            parallel_insert_start = time.time()

            # 4 core database, parallelism = 10 maxes out CPU
            await bounded_gather(
                *[functools.partial(process_chunk, chunk_counter, db, start_offset, end_offset, quiet=False)
                  for start_offset, end_offset in chunk_offsets],
                parallelism=10
            )
            print(f'took {time.time() - parallel_insert_start}s to insert the remaining complete records in parallel ({(chunk_size * len(chunk_offsets)) / (time.time() - parallel_insert_start)}) attempts / sec')

        print(f'finished populating records in {time.time() - populate_start_time}s')

        audit_start_time = time.time()
        await audit_changes(db)
        print(f'finished auditing changes in {time.time() - audit_start_time}')
    finally:
        print(f'finished migration in {time.time() - start_time}s')
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
