import secrets
import string
from datetime import datetime

from gear import Database


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def allocate_namespace(db: Database) -> str:
    namespace_name = f'test-ns-{generate_token()}'
    expiration = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    await db.execute_insertone(
        '''INSERT INTO internal_namespaces (`namespace_name`, `expiration_time`) VALUES (%s, %s)''',
        (namespace_name, expiration),
    )
    return namespace_name
