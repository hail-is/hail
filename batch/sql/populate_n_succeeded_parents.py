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


def offsets_to_where_statement(start_id, end_id):
    if start_id is None and end_id is None:
        where_cond = ''
        query_args = None
        return (where_cond, query_args)

    assert start_id != end_id, start_id
    if start_id is None:
        assert end_id
        end_batch_id, end_job_id = end_id
        where_cond = 'WHERE jobs.batch_id < %s OR ' \
                     '(jobs.batch_id = %s AND jobs.job_id < %s)'
        query_args = (end_batch_id, end_batch_id, end_job_id)
    elif end_id is None:
        assert start_id
        start_batch_id, start_job_id = start_id
        where_cond = 'WHERE jobs.batch_id > %s OR ' \
                     '(jobs.batch_id = %s AND jobs.job_id >= %s)'
        query_args = (start_batch_id, start_batch_id, start_job_id)
    else:
        assert start_id
        assert end_id
        start_batch_id, start_job_id = start_id
        end_batch_id, end_job_id = end_id
        where_cond = 'WHERE (jobs.batch_id > %s OR (jobs.batch_id = %s AND jobs.job_id >= %s)) ' \
                     'AND (jobs.batch_id < %s OR (jobs.batch_id = %s AND jobs.job_id < %s))'
        query_args = (start_batch_id, start_batch_id, start_job_id, end_batch_id, end_batch_id, end_job_id)

    return (where_cond, query_args)


async def process_chunk(counter, db, start_offset, end_offset, quiet=True):
    start_time = time.time()

    where_cond, query_args = offsets_to_where_statement(start_offset, end_offset)

    await db.just_execute(
        f'''
UPDATE jobs
SET migrated_n_succeeded_parents = TRUE
{where_cond}
''',
        query_args)

    if not quiet and counter.n % 100 == 0:
        print(f'processed chunk ({start_offset}, {end_offset}) in {time.time() - start_time}s')

    counter.n += 1
    if counter.n % 500 == 0:
        print(f'processed {counter.n} complete chunks')


async def find_chunk_offsets(db, size):
    @transaction(db)
    async def _find_chunks(tx) -> List[Optional[Tuple[int, int]]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        query = f'''
SELECT t.batch_id, t.job_id FROM (
  SELECT batch_id, job_id
  FROM jobs
  ORDER BY batch_id, job_id
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [(offset['batch_id'], offset['job_id']) async for offset in offsets]
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

        if chunk_offsets:
            print(f'found {len(chunk_offsets)} chunks to process')

            random.shuffle(chunk_offsets)

            burn_in_start = time.time()
            n_burn_in_chunks = 1000

            burn_in_chunk_offsets = chunk_offsets[:n_burn_in_chunks]
            chunk_offsets = chunk_offsets[n_burn_in_chunks:]

            for start_offset, end_offset in burn_in_chunk_offsets:
                await process_chunk(chunk_counter, db, start_offset, end_offset, quiet=False)

            print(f'finished burn-in in {time.time() - burn_in_start}s')

            parallel_update_start = time.time()

            await bounded_gather(
                *[functools.partial(process_chunk, chunk_counter, db, start_offset, end_offset, quiet=False)
                  for start_offset, end_offset in chunk_offsets],
                parallelism=6
            )
            print(f'took {time.time() - parallel_update_start}s to update the remaining records in parallel ({(chunk_size * len(chunk_offsets)) / (time.time() - parallel_update_start)}) attempts / sec')

        print(f'finished populating records in {time.time() - populate_start_time}s')
    finally:
        print(f'finished migration in {time.time() - start_time}s')
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
