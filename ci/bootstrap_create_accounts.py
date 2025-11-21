# ruff: noqa: E402
# pylint: disable=ungrouped-imports
from hailtop.hail_logging import configure_logging

configure_logging()

import base64
import json
import os
from typing import Optional

import kubernetes_asyncio.client
import kubernetes_asyncio.config

from auth.driver.driver import create_user
from gear import Database, Transaction, transaction
from gear.clients import get_identity_client
from gear.cloud_config import get_global_config
from hailtop.utils import async_to_blocking

CLOUD = get_global_config()['cloud']
SCOPE = os.environ['HAIL_SCOPE']
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']


async def insert_user_if_not_exists(app, username, login_id, system_roles, is_service_account):
    db = app['db']
    k8s_client = app['k8s_client']

    @transaction(db)
    async def insert(tx: Transaction) -> Optional[int]:
        row = await tx.execute_and_fetchone('SELECT id, state FROM users where username = %s;', (username,))
        if row:
            if row['state'] == 'active':
                return None
            if row['state'] == 'inactive':
                # Inactive users don't need recreating, but we should reactivate before we move on
                await tx.execute_update(
                    'UPDATE users SET state = "active", last_activated = CURRENT_TIMESTAMP(3) WHERE id = %s;',
                    (row['id'],),
                )
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

        # Technically we should be doing a permission check not a role check.
        # But since this is bootstrap we can assume the initial roles and permissions.
        # Also, all the users are hard coded below so the intention is not ambiguous.
        # Therefore: developer => access_developer_environments => give them a namespace
        if 'developer' in system_roles and SCOPE != 'deploy':
            namespace_name = DEFAULT_NAMESPACE
        else:
            namespace_name = None

        result = await tx.execute_insertone(
            """
    INSERT INTO users (state, username, login_id, is_service_account, hail_identity, hail_credentials_secret_name, namespace_name)
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    """,
            (
                'creating',
                username,
                login_id,
                is_service_account,
                hail_identity,
                hail_credentials_secret_name,
                namespace_name,
            ),
        )

        if result is not None and result > 0:
            for role in system_roles:
                await tx.execute_insertone(
                    """
    INSERT INTO users_system_roles (user_id, role_id)
    VALUES
    ((SELECT id FROM users WHERE username = '%s'), (SELECT id FROM system_roles WHERE name = '%s'));
    """,
                    (username, role),
                )
        return result

    return await insert()


async def main():
    users = [
        # username, login_id, system_roles, is_service_account
        ('auth', None, [], 1),
        ('batch', None, [], 1),
        ('ci', None, [], 1),
        ('test', None, [], 0),
        ('test-dev', None, ['developer', 'billing_manager', 'sysadmin'], 0),
        ('grafana', None, ['sysadmin-readonly'], 1),
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

        app['identity_client'] = get_identity_client()

        for username, login_id, system_roles, is_service_account in users:
            user_id = await insert_user_if_not_exists(app, username, login_id, system_roles, is_service_account)

            if user_id is not None:
                db_user = await db.execute_and_fetchone('SELECT * FROM users where id = %s;', (user_id,))
                db_user['system_roles'] = system_roles
                await create_user(app, db_user, skip_trial_bp=True)
    finally:
        await k8s_client.api_client.rest_client.pool_manager.close()


async_to_blocking(main())
