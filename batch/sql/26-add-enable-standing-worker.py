import os
import asyncio
from gear import Database, transaction


async def main():
    scope = os.environ['HAIL_SCOPE']
    enable_standing_worker = scope != 'dev'

    db = Database()
    await db.async_init()

    @transaction(db)
    async def insert_and_update(tx):
        await tx.just_execute('ALTER TABLE globals ADD COLUMN enable_standing_worker BOOLEAN NOT NULL DEFAULT FALSE;')
        if enable_standing_worker:
            await tx.execute_update('UPDATE globals SET enable_standing_worker = TRUE')
    await insert_and_update()  # pylint: disable=no-value-for-parameter

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
