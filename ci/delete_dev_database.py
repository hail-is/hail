import asyncio
import sys
from gear import Database

namespace = sys.argv[1]
database = sys.argv[2]
assert namespace != 'default'


async def main():
    db = Database()
    await db.async_init()
    await db.just_execute(f'DROP DATABASE `{namespace}-{database}`;')


asyncio.run(main())
