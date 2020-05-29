import os
import asyncio
from gear import Database, transaction


async def main():
    scope = os.environ['HAIL_SCOPE']

    if scope == 'deploy':
        standing_worker_cores = 4
    else:
        standing_worker_cores = 1

    db = Database()
    await db.async_init()

    @transaction(db)
    async def insert_and_update(tx):
        await tx.just_execute('ALTER TABLE globals ADD COLUMN standing_worker_cores BIGINT;')

        await tx.execute_update('UPDATE globals SET standing_worker_cores = %s',
                                (standing_worker_cores,))

        await tx.just_execute('ALTER TABLE globals MODIFY COLUMN standing_worker_cores BIGINT NOT NULL;')

    await insert_and_update()  # pylint: disable=no-value-for-parameter

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
