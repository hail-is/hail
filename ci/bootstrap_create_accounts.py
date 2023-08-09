from hailtop.hail_logging import configure_logging  # noqa: I001

configure_logging()

import base64  # noqa: E402 pylint: disable=wrong-import-position
import json  # noqa: E402 pylint: disable=wrong-import-position
import os  # noqa: E402 pylint: disable=wrong-import-position

import kubernetes_asyncio.client  # noqa: E402 pylint: disable=wrong-import-position
import kubernetes_asyncio.config  # noqa: E402 pylint: disable=wrong-import-position

from auth.driver.driver import create_user  # noqa: E402 pylint: disable=wrong-import-position
from gear import Database, transaction  # noqa: E402 pylint: disable=wrong-import-position
from gear.clients import get_identity_client  # noqa: E402 pylint: disable=wrong-import-position
from gear.cloud_config import get_global_config  # noqa: E402 pylint: disable=wrong-import-position
from hailtop.utils import async_to_blocking  # noqa: E402 pylint: disable=wrong-import-position,ungrouped-imports

CLOUD = get_global_config()['cloud']
SCOPE = os.environ['HAIL_SCOPE']
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']


async def insert_user_if_not_exists(app, username, login_id, is_developer, is_service_account):
    db = app['db']
    k8s_client = app['k8s_client']

    @transaction(db)
    async def insert(tx):
        row = await tx.execute_and_fetchone('SELECT id, state FROM users where username = %s;', (username,))
        if row:
            if row['state'] == 'active':
                return None
            return row['id']

        hail_credentials_secret_name = f'{username}-gsa-key'

        secret = await k8s_client.read_namespaced_secret(hail_credentials_secret_name, DEFAULT_NAMESPACE)
        credentials_json = base64.b64decode(secret.data['key.json']).decode()
        credentials = json.loads(credentials_json)

        if CLOUD == 'gcp':
            hail_identity = credentials['client_email']
        else:
            assert CLOUD == 'azure'
            hail_identity = credentials['appObjectId']

        if is_developer and SCOPE != 'deploy':
            namespace_name = DEFAULT_NAMESPACE
        else:
            namespace_name = None

        return await tx.execute_insertone(
            '''
    INSERT INTO users (state, username, login_id, is_developer, is_service_account, hail_identity, hail_credentials_secret_name, namespace_name)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    ''',
            (
                'creating',
                username,
                login_id,
                is_developer,
                is_service_account,
                hail_identity,
                hail_credentials_secret_name,
                namespace_name,
            ),
        )

    return await insert()


async def main():
    users = [
        # username, login_id, is_developer, is_service_account
        ('auth', None, 0, 1),
        ('ci', None, 0, 1),
        ('test', None, 0, 0),
        ('test-dev', None, 1, 0),
        ('grafana', None, 0, 1),
    ]

    app = {}

    db = Database()
    await db.async_init(maxsize=50)
    app['db'] = db

    db_instance = Database()
    await db_instance.async_init(maxsize=50, config_file='/database-server-config/sql-config.json')
    app['db_instance'] = db_instance

    # kube.config.load_incluster_config()
    await kubernetes_asyncio.config.load_kube_config()
    k8s_client = kubernetes_asyncio.client.CoreV1Api()
    try:
        app['k8s_client'] = k8s_client

        app['identity_client'] = get_identity_client(credentials_file='/auth-gsa-key/key.json')

        for username, login_id, is_developer, is_service_account in users:
            user_id = await insert_user_if_not_exists(app, username, login_id, is_developer, is_service_account)

            if user_id is not None:
                db_user = await db.execute_and_fetchone('SELECT * FROM users where id = %s;', (user_id,))
                await create_user(app, db_user, skip_trial_bp=True)
    finally:
        await k8s_client.api_client.rest_client.pool_manager.close()


async_to_blocking(main())
