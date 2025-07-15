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


async def process_chunk_attempt_resources(counter, db, start, end, quiet=True):
    if start is None:
        assert end is not None
        end_batch_id, end_job_id, end_attempt_id, end_resource_id = end

        where_statement = '''
WHERE (attempt_resources.batch_id < %s) OR
      (attempt_resources.batch_id = %s AND attempt_resources.job_id < %s) OR
      (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id < %s) OR
      (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id = %s AND attempt_resources.resource_id < %s)
'''
        query_args = [end_batch_id,
                      end_batch_id, end_job_id,
                      end_batch_id, end_job_id, end_attempt_id,
                      end_batch_id, end_job_id, end_attempt_id, end_resource_id,
                      ]
    elif end is None:
        assert start is not None
        start_batch_id, start_job_id, start_attempt_id, start_resource_id = start

        where_statement = '''
WHERE (attempt_resources.batch_id > %s) OR
      (attempt_resources.batch_id = %s AND attempt_resources.job_id > %s) OR
      (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id > %s) OR
      (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id = %s AND attempt_resources.resource_id >= %s)
'''
        query_args = [start_batch_id,
                      start_batch_id, start_job_id,
                      start_batch_id, start_job_id, start_attempt_id,
                      start_batch_id, start_job_id, start_attempt_id, start_resource_id,
                      ]
    else:
        assert start is not None and end is not None
        start_batch_id, start_job_id, start_attempt_id, start_resource_id = start
        end_batch_id, end_job_id, end_attempt_id, end_resource_id = end

        where_statement = '''
WHERE ((attempt_resources.batch_id > %s) OR
       (attempt_resources.batch_id = %s AND attempt_resources.job_id > %s) OR
       (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id > %s) OR
       (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id = %s AND attempt_resources.resource_id >= %s))
  AND ((attempt_resources.batch_id < %s) OR
       (attempt_resources.batch_id = %s AND attempt_resources.job_id < %s) OR
       (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id < %s) OR
       (attempt_resources.batch_id = %s AND attempt_resources.job_id = %s AND attempt_resources.attempt_id = %s AND attempt_resources.resource_id < %s))
'''
        query_args = [start_batch_id,
                      start_batch_id, start_job_id,
                      start_batch_id, start_job_id, start_attempt_id,
                      start_batch_id, start_job_id, start_attempt_id, start_resource_id,
                      end_batch_id,
                      end_batch_id, end_job_id,
                      end_batch_id, end_job_id, end_attempt_id,
                      end_batch_id, end_job_id, end_attempt_id, end_resource_id,
                      ]

    query = f'''
UPDATE attempt_resources
LEFT JOIN resources ON attempt_resources.resource_id = resources.resource_id
SET attempt_resources.deduped_resource_id = resources.deduped_resource_id
{where_statement}
'''

    await process_chunk(counter, db, query, query_args, start, end, quiet)


async def audit_changes(db):
    audit_start = time.time()
    print('starting auditing attempt resources')

    bad_attempt_resources_records = db.select_and_fetchall(
        '''
SELECT * FROM attempt_resources
WHERE deduped_resource_id IS NULL
LIMIT 100;
'''
    )

    print(f'finished auditing attempt resources records in {time.time() - audit_start}s')

    bad_attempt_resources_records = [record async for record in bad_attempt_resources_records]
    for record in bad_attempt_resources_records:
        print(f'found bad record {record}')

    if bad_attempt_resources_records:
        raise Exception(f'errors found in audit')


async def find_chunk_offsets(db, size):
    @transaction(db)
    async def _find_chunks(tx) -> List[Optional[Tuple[int, int, str]]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        query = f'''
SELECT t.batch_id, t.job_id, t.attempt_id, t.resource_id FROM (
  SELECT batch_id, job_id, attempt_id, resource_id
  FROM attempt_resources
  ORDER BY batch_id ASC, job_id ASC, attempt_id ASC, resource_id ASC
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [(offset['batch_id'], offset['job_id'], offset['attempt_id'], offset['resource_id']) async for offset in offsets]
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
            await process_chunk_attempt_resources(chunk_counter, db, start_offset, end_offset, quiet=False)

        print(f'finished burn-in in {time.time() - burn_in_start}s')

        parallel_insert_start = time.time()

        # 4 core database, parallelism = 10 maxes out CPU
        await bounded_gather(
            *[functools.partial(process_chunk_attempt_resources, chunk_counter, db, start_offset, end_offset, quiet=False)
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
