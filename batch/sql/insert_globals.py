import os
import string
import secrets
import asyncio
from gear import Database


async def main():
    scope = os.environ['HAIL_SCOPE']

    worker_type = 'standard'
    if scope == 'deploy':
        worker_cores = 16
        worker_disk_size_gb = 100
        max_instances = 10
        pool_size = 10
    else:
        worker_cores = 1
        worker_disk_size_gb = 10
        max_instances = 3
        pool_size = 2
        
    db = Database()
    await db.async_init()

    # 22 is 16 bytes of entropy
    # math.log((2**8)**16, 62) = 21.497443706501368
    sigma = string.ascii_letters + string.digits
    instance_id = ''.join([secrets.choice(sigma) for _ in range(22)])

    await db.execute_insertone(
        '''
INSERT INTO globals (
  instance_id, internal_token,
  worker_cores, worker_type, worker_disk_size_gb, max_instances, pool_size)
VALUES (%s, %s, %s, %s, %s, %s, %s);
''',
        (instance_id, secrets.token_urlsafe(32),
         worker_cores, worker_type, worker_disk_size_gb, max_instances, pool_size))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
