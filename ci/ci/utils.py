import datetime
import json
import os
import random
import secrets
import string
from typing import Dict, List, Optional, Set

from gear import Database, transaction

TEST_OAUTH2_CALLBACK_URLS: Set[str] = set(json.loads(os.environ.get('HAIL_TEST_OAUTH2_CALLBACK_URLS', '[]')))


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


async def reserve_namespace(
    db: Database,
    namespace: str,
    expiration_time: Optional[datetime.datetime] = None,
) -> str:
    if namespace == 'default':
        raise ValueError('Cannot reserve the default namespace')

    @transaction(db)
    async def reserve(tx) -> str:
        in_use: Dict[str, str] = {
            record['namespace']: record['oauth2_callback_url']
            async for record in tx.execute_and_fetchall(
                'SELECT namespace, oauth2_callback_url FROM active_namespaces FOR UPDATE'
            )
        }
        # This namespace might already be deployed
        callback_url: str = in_use.get(namespace) or random.choice(
            list(TEST_OAUTH2_CALLBACK_URLS.difference(in_use.values()))
        )

        expiration = expiration_time.strftime('%Y-%m-%d %H:%M:%S') if expiration_time else None
        await tx.execute_insertone(
            '''
INSERT INTO active_namespaces (`namespace`, `oauth2_callback_url`, `expiration_time`)
VALUES (%s, %s, %s) as new_ns
ON DUPLICATE KEY UPDATE expiration_time = new_ns.expiration_time, oauth2_callback_url = new_ns.oauth2_callback_url
''',
            (namespace, callback_url, expiration),
        )

        return callback_url

    return await reserve()  # pylint: disable=no-value-for-parameter


async def release_namespace(db: Database, namespace: str):
    assert namespace != 'default'
    await db.just_execute('DELETE FROM active_namespaces WHERE namespace = %s', (namespace,))


async def add_deployed_services(
    db: Database,
    namespace: str,
    services: List[str],
):
    await db.execute_many(
        '''
INSERT INTO deployed_services (`namespace`, `service`)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE namespace = namespace;
''',
        [(namespace, service) for service in services],
    )
