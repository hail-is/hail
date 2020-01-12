import asyncio
from gear import Database


async def main():
    db = Database()
    await db.async_init()

    await db.just_execute('CALL insert_ready_cores_tokens();')

    users = db.execute_and_fetchall('SELECT user FROM user_resources;')
    async for user in users:
        await db.just_execute('CALL insert_user_resources_tokens(%s)',
                              (user,))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
