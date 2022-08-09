import secrets
import string
import json
import datetime

from typing import List

from gear import Database


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def reserve_namespace(db: Database, namespace_name: str, services: List[str]):
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    expiration = tomorrow.strftime('%Y-%m-%d %H:%M:%S')
    await db.execute_insertone(
        '''INSERT INTO internal_namespaces (`namespace_name`, `expiration_time`, `services`) VALUES (%s, %s, %s)''',
        (namespace_name, expiration, json.dumps(services)),
    )
