import asyncio
from gear import Database


async def async_main():
    db = Database()
    await db.async_init()

    await db.just_execute(
        f"INSERT INTO hello2 (greeting) VALUES ('hello, hello!');")


loop = asyncio.get_event_loop()
loop.run_until_complete(async_main())
