import string
import secrets
import asyncio
from gear import Database

async def main():
    db = Database()
    await db.async_init()

    # 22 is 16 bytes of entropy
    # math.log((2**8)**16, 62) = 21.497443706501368
    sigma = string.ascii_letters + string.digits
    instance_id = ''.join([secrets.choice(sigma) for _ in range(22)])
    await db.execute_insertone(
        'INSERT INTO tokens (name, token) VALUES (%s, %s);',
        ('instance_id', instance_id))

    # for securing communication between front end and driver
    await db.execute_insertone(
        'INSERT INTO tokens (name, token) VALUES (%s, %s);',
        ('internal', secrets.token_urlsafe(32)))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
