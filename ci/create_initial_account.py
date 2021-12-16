import argparse
import base64
import json
import kubernetes_asyncio as kube
import os

from hailtop.utils import async_to_blocking
from gear import Database, transaction


NAMESPACE = os.environ['NAMESPACE']


async def copy_identity_from_default(hail_credentials_secret_name: str) -> str:
    cloud = os.environ['CLOUD']
    await kube.config.load_kube_config()
    k8s_client = kube.client.CoreV1Api()

    secret = await k8s_client.read_namespaced_secret(hail_credentials_secret_name, 'default')

    await k8s_client.create_namespaced_secret(
        NAMESPACE,
        kube.client.V1Secret(
            metadata=kube.client.V1ObjectMeta(name=hail_credentials_secret_name),
            data=secret.data,
        ),
    )

    credentials_json = base64.b64decode(secret.data['key.json']).decode()
    credentials = json.loads(credentials_json)

    if cloud == 'gcp':
        return credentials['client_email']
    assert cloud == 'azure'
    return credentials['appObjectId']


async def insert_user_if_not_exists(db, username, login_id, is_developer, is_service_account):
    @transaction(db)
    async def insert(tx):
        row = await db.execute_and_fetchone('SELECT id, state FROM users where username = %s;', (username,))
        if row:
            if row['state'] == 'active':
                return None
            return row['id']

        if NAMESPACE == 'default':
            hail_credentials_secret_name = None
            hail_identity = None
            namespace_name = None
        else:
            hail_credentials_secret_name = f'{username}-gsa-key'
            hail_identity = await copy_identity_from_default(hail_credentials_secret_name)
            namespace_name = NAMESPACE

        return await db.execute_insertone(
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
    parser = argparse.ArgumentParser(description='Create an initial dev user.')

    parser.add_argument('username', help='The username of the initial user.')
    parser.add_argument('login_id', metavar='login-id', help='The login id of the initial user.')

    args = parser.parse_args()

    db = Database()
    await db.async_init(maxsize=50)

    await insert_user_if_not_exists(db, args.username, args.login_id, True, False)


async_to_blocking(main())
