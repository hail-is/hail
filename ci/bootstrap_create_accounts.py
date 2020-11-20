import os
import kubernetes_asyncio as kube
from hailtop import aiogoogle
from hailtop import batch_client as bc
from hailtop.utils import async_to_blocking
from gear import Database

from auth.driver.driver import create_user

PROJECT = os.environ['HAIL_PROJECT']
GSA_EMAIL = os.environ.get('HAIL_GSA_EMAIL')
NAMESPACE_NAME = os.environ.get('HAIL_NAMESPACE_NAME')


async def insert_user_if_not_exists(db, username, email, is_developer, is_service_account):
    row = await db.execute_and_fetchone('SELECT id FROM users where username = %s;', (username,))
    if row:
        return row['id']

    # If scope = test, dev we can't create our own service accounts or
    # namespaces.  Use the ones given to us.
    gsa_email = GSA_EMAIL
    if gsa_email:
        gsa_key_secret_name = f'{username}-gsa-key'
    else:
        gsa_key_secret_name = None

    namespace_name = NAMESPACE_NAME

    return await db.execute_insertone(
        '''
INSERT INTO users (state, username, email, is_developer, is_service_account, gsa_email, gsa_key_secret_name, namespace_name)
VALUES (%s, %s, %s, %s, %s);
''',
        ('creating', username, email, is_developer, is_service_account, gsa_email, gsa_key_secret_name, namespace_name))


async def main():
    users = [
        # username, email, is_developer, is_service_account
        ('auth', None, 0, 1),
        ('benchmark', None, 0, 1),
        ('ci', None, 0, 1),
        ('test', None, 0, 0),
        ('test-dev', None, 1, 0)
    ]

    app = {}

    db = Database()
    await db.async_init(maxsize=50)
    app['db'] = db

    db_instance = Database()
    await db_instance.async_init(maxsize=50, config_file='/database-server-config/sql-config.json')
    app['db_instance'] = db_instance

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['k8s_client'] = k8s_client

    app['iam_client'] = aiogoogle.IAmClient(
        PROJECT, credentials=aiogoogle.Credentials.from_file('/gsa-key/key.json'))

    app['batch_client'] = await bc.aioclient.BatchClient(None)

    for username, email, is_developer, is_service_account in users:
        user_id = await insert_user_if_not_exists(db, username, email, is_developer, is_service_account)

    db_user = await db.execute_and_fetchone('SELECT * FROM users where id = %s;', (user_id,))
    await create_user(app, db_user)


async_to_blocking(main())
