import asyncio
import functools
import os
import random
import time
from typing import List, Optional

from gear import Database, transaction
from hailtop.utils import bounded_gather


MYSQL_CONFIG_FILE = os.environ.get('MYSQL_CONFIG_FILE')


class Counter:
    def __init__(self):
        self.n = 0


def offsets_to_where_statement(start_batch_id, end_batch_id):
    if start_batch_id is None and end_batch_id is None:
        where_cond = ''
        query_args = None
        return (where_cond, query_args)

    assert start_batch_id != end_batch_id, start_batch_id
    if start_batch_id is None:
        assert end_batch_id
        where_cond = 'WHERE batches.id < %s'
        query_args = (end_batch_id,)
    elif end_batch_id is None:
        assert start_batch_id
        where_cond = 'WHERE batches.id >= %s'
        query_args = (start_batch_id,)
    else:
        where_cond = 'WHERE batches.id >= %s AND batches.id < %s'
        query_args = (start_batch_id, end_batch_id)

    return (where_cond, query_args)


async def process_chunk(counter, db, start_offset, end_offset, quiet=True):
    start_time = time.time()

    where_cond, query_args = offsets_to_where_statement(start_offset, end_offset)

    await db.just_execute(
        f'''
UPDATE batches
SET migrated_batch = 1
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
    async def _find_chunks(tx) -> List[Optional[int]]:
        start_time = time.time()

        await tx.just_execute('SET @rank=0;')

        query = f'''
SELECT t.id FROM (
  SELECT id
  FROM batches
  ORDER BY id
) AS t
WHERE MOD((@rank := @rank + 1), %s) = 0;
'''

        offsets = tx.execute_and_fetchall(query, (size,))
        offsets = [offset['id'] async for offset in offsets]
        offsets.append(None)

        print(f'found chunk offsets in {round(time.time() - start_time, 4)}s')
        return offsets

    return await _find_chunks()


async def audit(db):
    bad_records = db.execute_and_fetchall(
        '''
SELECT id AS batch_id
FROM batches
LEFT JOIN job_groups ON batches.id = job_groups.batch_id
WHERE job_groups.batch_id IS NULL
LIMIT 100;
''')

    bad_batch_ids = [record['batch_id'] async for record in bad_records]

    if bad_batch_ids:
        print(f'missing records in job groups: {bad_batch_ids}')
        raise Exception('audit failed. missing records in job_groups')

    bad_parent_records = db.execute_and_fetchall(
        '''
SELECT id AS batch_id
FROM batches
LEFT JOIN job_group_parents ON batches.id = job_group_parents.batch_id
WHERE job_groups.batch_id IS NULL
LIMIT 100;
''')

    bad_batch_ids = [record['batch_id'] async for record in bad_parent_records]

    if bad_batch_ids:
        print(f'missing records in job group_parents: {bad_batch_ids}')
        raise Exception('audit failed. missing records in job_group_parents')


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

        await audit(db)

        print(f'finished populating records in {time.time() - populate_start_time}s')
    finally:
        print(f'finished migration in {time.time() - start_time}s')
        await db.async_close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
