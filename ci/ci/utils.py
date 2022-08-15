import datetime
import secrets
import string
from typing import Dict, List, Optional

from gear import Database


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def add_deployed_services(
    db: Database,
    services_per_namespace: Dict[str, List[str]],
    expiration_time: Optional[datetime.datetime],
):
    expiration = expiration_time.strftime('%Y-%m-%d %H:%M:%S') if expiration_time else None
    for namespace, services in services_per_namespace.items():
        await db.execute_insertone(
            '''
INSERT INTO active_namespaces (`namespace`, `expiration_time`)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE expiration_time = VALUES(expiration_time)
            ''',
            (namespace, expiration),
        )
        await db.execute_many(
            'INSERT IGNORE INTO deployed_services (`namespace`, `service`) VALUES (%s, %s)',
            [(namespace, service) for service in services],
        )
