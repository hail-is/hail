import asyncio
import functools
import os
import random
import time

from gear import Database, transaction
from hailtop.utils import bounded_gather, secret_alnum_string

MYSQL_CONFIG_FILE = os.environ.get('MYSQL_CONFIG_FILE')


class Counter:
    def __init__(self):
        self.n = 0


async def process_chunk(counter, db, start_offset, end_offset, size=None, quiet=True):
    # start inclusive
    # end exclusive

    @transaction(db)
    async def _process(tx):
        start_time = time.time()

        assert start_offset != end_offset, start_offset

        if start_offset is None:
            start_batch_id, start_job_id, start_attempt_id = None, None, None
        else:
            start_batch_id, start_job_id, start_attempt_id = start_offset

        if end_offset is None:
            end_batch_id, end_job_id, end_attempt_id = None, None, None
        else:
            end_batch_id, end_job_id, end_attempt_id = end_offset

        resources = [
            ('compute/n1-preemptible/1', 'cores_mcpu'),
            ('memory/n1-preemptible/1', '3840 * (cores_mcpu DIV 1000)'),  # standard worker has 3840 mi per core
            ('boot-disk/pd-ssd/1', '100 * 1024 * cores_mcpu DIV (16 * 1000)'),  # worker fraction assumes there are 16 cores and 100 gi of disk
            ('ip-fee/1024/1', '1024 * cores_mcpu DIV (16 * 1000)'),
            ('service-fee/1', 'cores_mcpu'),
        ]

        for resource, quantity in resources:
            if start_batch_id is None or start_job_id is None or start_attempt_id is None:
                assert end_batch_id
                where_cond = 'WHERE batches.`format_version` < 3 AND attempts.batch_id < %s OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id < %s) OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id < %s)'
                query_args = (resource, end_batch_id, end_batch_id, end_job_id, end_batch_id, end_job_id, end_attempt_id)
            elif end_batch_id is None or end_job_id is None or end_attempt_id is None:
                assert start_batch_id
                where_cond = 'WHERE batches.`format_version` < 3 AND ' \
                             'attempts.batch_id > %s OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id > %s) OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id >= %s)'
                query_args = (resource, start_batch_id, start_batch_id, start_job_id, start_batch_id, start_job_id, start_attempt_id)
            else:
                # adding a where check for format_version < 3 here causes a full table scan
                where_cond = 'WHERE (attempts.batch_id > %s OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id > %s) OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id >= %s)) ' \
                             'AND (attempts.batch_id < %s OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id < %s) OR ' \
                             '(attempts.batch_id = %s AND attempts.job_id = %s AND attempts.attempt_id < %s))'
                query_args = (resource,
                              start_batch_id, start_batch_id, start_job_id, start_batch_id, start_job_id, start_attempt_id,
                              end_batch_id, end_batch_id, end_job_id, end_batch_id, end_job_id, end_attempt_id)

            if size is not None:
                limit = f'LIMIT {size}'
            else:
                limit = ''

            query = f'''
INSERT INTO attempt_resources (batch_id, job_id, attempt_id, resource_id, quantity)
SELECT * FROM (
  SELECT attempts.batch_id, attempts.job_id, attempts.attempt_id, (
    SELECT resource_id
    FROM resources
    WHERE resource = %s
  ), {quantity}
  FROM attempts FORCE INDEX (PRIMARY)
  LEFT JOIN jobs ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
  LEFT JOIN batches ON attempts.batch_id = batches.id
  {where_cond}
  {limit}
) AS t
ON DUPLICATE KEY UPDATE quantity = quantity;
'''

            await tx.just_execute(query, query_args)

        if not quiet:
            print(f'processed chunk ({start_offset}, {end_offset}) in {time.time() - start_time}s')

        counter.n += 1
        if counter.n % 500 == 0:
            print(f'processed {counter.n} chunks')

    return await _process()


async def audit_changes(db, expected_n_inserts):
    result = await db.select_and_fetchone(
        '''
SELECT COUNT(*) as count
FROM attempt_resources
LEFT JOIN batches ON batches.id = attempt_resources.batch_id
WHERE batches.`format_version` < 3;
'''
    )

    if result['count'] != expected_n_inserts:
        raise Exception(
            f'number of attempt resources inserted ({result["count"]})does not match expected value {expected_n_inserts}')

    bad_job_records = db.select_and_fetchall(
        '''
SELECT old.batch_id, old.job_id, old.cost, new.cost, ABS(new.cost - old.cost) AS cost_diff
FROM (
  SELECT batch_id, job_id, (jobs.msec_mcpu * 0.001 * 0.001) * ((0.01 + ((0.17 * 100 / 30.4375 / 24 + 0.004) / 16) + 0.01) / 3600) AS cost
  FROM jobs
  LEFT JOIN batches ON jobs.batch_id = batches.id
  WHERE batches.`format_version` < 3) AS old
LEFT JOIN (
  SELECT attempt_resources.batch_id, attempt_resources.job_id, COALESCE(SUM(GREATEST(COALESCE(end_time - start_time, 0), 0) * quantity * rate), 0) AS cost
  FROM attempt_resources
  LEFT JOIN batches ON attempt_resources.batch_id = batches.id
  LEFT JOIN attempts ON attempt_resources.batch_id = attempts.batch_id AND attempt_resources.job_id = attempts.job_id AND attempt_resources.attempt_id = attempts.attempt_id
  LEFT JOIN resources ON attempt_resources.resource_id = resources.resource_id
  WHERE batches.`format_version` < 3
  GROUP BY batch_id, job_id
) AS new ON new.batch_id = old.batch_id AND new.job_id = old.job_id
WHERE COALESCE(ABS(new.cost - old.cost), -1) >= 0.001
LIMIT 100;
''')

    async for record in bad_job_records:
        raise Exception(f'found bad record {record}')


async def find_chunk_offsets(db, size):
    @transaction(db)
    async def _find_chunks(tx):
        start_time = time.time()

        await tx.just_execute('SET @rank = 0;')

        query = f'''
SELECT t.batch_id, t.job_id, t.attempt_id FROM (
  SELECT attempts.batch_id, attempts.job_id, attempts.attempt_id
  FROM attempts
  LEFT JOIN batches ON attempts.batch_id = batches.id
  WHERE batches.`format_version` < 3
  ORDER BY attempts.batch_id, attempts.job_id, attempts.attempt_id
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [(offset['batch_id'], offset['job_id'], offset['attempt_id']) async for offset in offsets]

        last_offset = tx.execute_and_fetchall(
            '''
SELECT batch_id, job_id, attempt_id
FROM attempts
LEFT JOIN batches ON attempts.batch_id = batches.id
WHERE batches.format_version >= 3
ORDER BY attempts.batch_id, attempts.job_id, attempts.attempt_id
LIMIT 1;
'''
        )
        last_offset = [(offset['batch_id'], offset['job_id'], offset['attempt_id']) async for offset in last_offset]
        assert len(last_offset) == 1, last_offset
        last_offset = last_offset[0]
        offsets.append(last_offset)

        print(f'found chunk offsets in {round(time.time() - start_time, 4)}s')
        return offsets

    return await _find_chunks()


async def main(chunk_size=100):
    db = Database()
    await db.async_init(config_file=MYSQL_CONFIG_FILE)

    start_time = time.time()
    counter = Counter()

    try:
        count = await db.select_and_fetchone(
            '''
SELECT COUNT(*) as count FROM attempts
LEFT JOIN batches ON attempts.batch_id = batches.id
WHERE batches.`format_version` < 3;
'''
        )
        n_attempts_expected = count['count']
        print(f'expecting to process {n_attempts_expected} attempts')

        if n_attempts_expected > 0:
            chunk_offsets = [None]
            for offset in await find_chunk_offsets(db, chunk_size):
                chunk_offsets.append(offset)

            # the endpoint of the last offset is the first record where format version >= 3
            # thus, the start_offset should never equal the end offset
            chunk_offsets = list(zip(chunk_offsets[:-1], chunk_offsets[1:]))

            print(f'found {len(chunk_offsets)} chunks to process')

            random.shuffle(chunk_offsets)

            n_burn_in_chunks = 5000
            burn_in_chunk_size = 10

            start_insert = time.time()

            burn_in_start = time.time()
            burn_in_chunk_offsets = chunk_offsets[:n_burn_in_chunks]
            for start_offset, end_offset in burn_in_chunk_offsets:
                await process_chunk(counter, db, start_offset, end_offset, size=burn_in_chunk_size)

            print(f'finished burn-in in {time.time() - burn_in_start}s')

            parallel_insert_start = time.time()

            # 4 core database, parallelism = 10 maxes out CPU
            await bounded_gather(
                *[functools.partial(process_chunk, counter, db, start_offset, end_offset, quiet=True) for start_offset, end_offset in chunk_offsets],
                parallelism=10
            )
            print(f'took {time.time() - parallel_insert_start}s to insert the remaining records in parallel ({(chunk_size * len(chunk_offsets)) / (time.time() - parallel_insert_start)}) attempts / sec')

            print(f'took {time.time() - start_insert}s to insert all records')

            await audit_changes(db, n_attempts_expected * 5)
    finally:
        print(f'finished migration in {time.time() - start_time}s')
        await db.async_close()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
