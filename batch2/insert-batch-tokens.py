import secrets
import asyncio
from gear import Database

async def main():
    db = Database()
    await db.async_init()

    instance_id = secrets.token_urlsafe(16)
    instance_id = ''.join([c for c in instance_id if c not in '_-'])
    await db.execute_insertone(
        'INSERT INTO tokens (name, token) VALUES (%s, %s);',
        'instance_id', instance_id)

    # for securing communication between front end and driver
    await db.execute_insertone(
        'INSERT INTO tokens (name, token) VALUES (%s, %s);',
        'internal', secrets.token_urlsafe(32))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
