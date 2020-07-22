import os
import random
import json
import base64
import logging
import secrets
import string
import asyncio
import aiohttp
import kubernetes_asyncio as kube
from hailtop.utils import time_msecs
from hailtop.auth.sql_config import create_secret_data_from_config
from hailtop import aiogoogle
from gear import create_session, Database

log = logging.getLogger('auth.driver')

PROJECT = os.environ['PROJECT']
ZONE = os.environ['ZONE']
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
BATCH_PODS_NAMESPACE = os.environ['HAIL_BATCH_PODS_NAMESPACE']

is_test_deployment = DEFAULT_NAMESPACE != 'default'


class DatabaseConflictError(Exception):
    pass


class EventHandler:
    def __init__(self, handler, event=None, bump_secs=60.0, min_delay_secs=0.1):
        self.handler = handler
        if event is None:
            event = asyncio.Event()
        self.event = event
        self.bump_secs = bump_secs
        self.min_delay_secs = min_delay_secs

    async def main_loop(self):
        delay_secs = self.min_delay_secs
        while True:
            try:
                start_time = time_msecs()
                while True:
                    self.event.clear()
                    should_wait = await self.handler()
                    if should_wait:
                        await self.event.wait()
            except Exception:
                end_time = time_msecs()

                log.exception('caught exception in event handler loop')

                t = delay_secs * random.uniform(0.7, 1.3)
                await asyncio.sleep(t)

                ran_for_secs = (end_time - start_time) * 1000
                delay_secs = min(
                    max(self.min_delay_secs, 2 * delay_secs - min(0, (ran_for_secs - t) / 2)),
                    30.0)

    async def bump_loop(self):
        while True:
            try:
                self.event.set()
            except Exception:
                log.exception('caught exception')
            await asyncio.sleep(self.bump_secs)

    async def start(self):
        asyncio.ensure_future(self.main_loop())
        if self.bump_secs is not None:
            asyncio.ensure_future(self.bump_loop())


class SessionResource:
    def __init__(self, db, session_id=None):
        self.db = db
        self.session_id = session_id

    async def create(self, user_id, *args, **kwargs):
        self.session_id = await create_session(self.db, user_id, *args, **kwargs)

    async def delete(self):
        if self.session_id is None:
            return

        await self.db.just_execute('''
DELETE FROM sessions
WHERE session_id = %s;
''',
                                   (self.session_id,))
        self.session_id = None


class K8sSecretResource:
    def __init__(self, k8s_client, name=None, namespace=None):
        self.k8s_client = k8s_client
        self.name = name
        self.namespace = namespace

    async def create(self, name, namespace, data):
        assert self.name is None and self.namespace is None

        await self._delete(name, namespace)

        await self.k8s_client.create_namespaced_secret(
            namespace,
            kube.client.V1Secret(
                metadata=kube.client.V1ObjectMeta(
                    name=name),
                data={
                    k: base64.b64encode(v.encode('utf-8')).decode('utf-8')
                    for k, v in data.items()
                }))
        self.name = name

    async def _delete(self, name, namespace):
        try:
            await self.k8s_client.delete_namespaced_secret(
                name, namespace)
        except kube.client.rest.ApiException as e:
            if e.status == 404:
                pass
            else:
                raise

    async def delete(self):
        if self.name is None:
            return
        await self._delete(self.name, self.namespace)
        self.name = None
        self.namespace = None


class GSAResource:
    def __init__(self, iam_client, gsa_email=None):
        self.iam_client = iam_client
        self.gsa_email = gsa_email

    async def create(self, username):
        assert self.gsa_email is None

        gsa_email = f'{username}@{PROJECT}.iam.gserviceaccount.com'

        await self._delete(gsa_email)

        service_account = await self.iam_client.post(
            '/serviceAccounts',
            json={
                "accountId": username,
                "serviceAccount": {
                    "displayName": username
                }
            })
        assert service_account['email'] == gsa_email
        self.gsa_email = gsa_email

        key = await self.iam_client.post(
            f'/serviceAccounts/{self.gsa_email}/keys')

        return (self.gsa_email, key)

    async def _delete(self, gsa_email):
        try:
            await self.iam_client.delete(f'/serviceAccounts/{gsa_email}/keys')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                pass
            else:
                raise

    async def delete(self):
        if self.gsa_email is None:
            return
        await self._delete(self.gsa_email)
        self.gsa_email = None


class DatabaseResource:
    def __init__(self, db_instance, name=None):
        self.db_instance = db_instance
        self.name = name
        self.password = None

    async def create(self, name):
        assert self.name is None

        if is_test_deployment:
            return

        await self._delete(name)

        self.password = secrets.token_urlsafe(16)
        await self.db_instance.just_execute(f'''
CREATE DATABASE `{name}`;

CREATE USER '{name}'@'%' IDENTIFIED BY '{self.password}';
GRANT ALL ON `{name}`.* TO '{name}'@'%';
''')
        self.name = name

    def secret_data(self):
        with open('/database-server-config/sql-config.json', 'r') as f:
            server_config = json.loads(f.read())
        with open('/database-server-config/server-ca.pem', 'r') as f:
            server_ca = f.read()
        with open('/database-server-config/client-cert.pem', 'r') as f:
            client_cert = f.read()
        with open('/database-server-config/client-key.pem', 'r') as f:
            client_key = f.read()

        if is_test_deployment:
            return create_secret_data_from_config(
                server_config, server_ca, client_cert, client_key)

        assert self.name is not None
        assert self.password is not None

        config = {
            'host': server_config['host'],
            'port': server_config['port'],
            'user': self.name,
            'password': self.password,
            'instance': server_config['instance'],
            'connection_name': server_config['connection_name'],
            'db': self.name
        }
        return create_secret_data_from_config(
            config, server_ca, client_cert, client_key)

    async def _delete(self, name):
        if is_test_deployment:
            return

        # no DROP USER IF EXISTS in current db version
        row = await self.db_instance.execute_and_fetchone(
            'SELECT 1 FROM mysql.user WHERE User = %s;', (name,))
        if row is not None:
            await self.db_instance.just_execute(f"DROP USER '{name}';")

        await self.db_instance.just_execute(f'DROP DATABASE IF EXISTS `{name}`;')

    async def delete(self):
        if self.name is None:
            return
        await self._delete(self.name)
        self.name = None


class K8sNamespaceResource:
    def __init__(self, k8s_client, name=None):
        self.k8s_client = k8s_client
        self.name = name

    async def create(self, name):
        assert self.name is None

        await self._delete(name)

        await self.k8s_client.create_namespace(
            kube.client.V1Namespace(
                metadata=kube.client.V1ObjectMeta(
                    name=name)))
        self.name = name

    async def _delete(self, name):
        try:
            await self.k8s_client.delete_namespace(name)
        except kube.client.rest.ApiException as e:
            if e.status == 404:
                pass
            else:
                raise

    async def delete(self):
        if self.name is None:
            return
        await self._delete(self.name)
        self.name = None


async def _create_user(app, user, cleanup):
    db_instance = app['db_instance']
    db = app['db']
    k8s_client = app['k8s_client']
    iam_client = app['iam_client']

    alnum = string.ascii_lowercase + string.digits
    token = ''.join([secrets.choice(alnum) for _ in range(5)])
    ident_token = f'{user["username"]}-{token}'
    if user['is_developer'] == 1:
        ident = user['username']
    else:
        ident = ident_token

    updates = {
        'state': 'active'
    }

    tokens_secret_name = user['tokens_secret_name']
    if tokens_secret_name is None:
        tokens_session = SessionResource(db)
        cleanup.append(tokens_session.delete)
        await tokens_session.create(user['id'], max_age_secs=None)

        tokens_secret_name = f'{ident}-tokens'
        tokens_secret = K8sSecretResource(k8s_client)
        cleanup.append(tokens_secret.delete)
        await tokens_secret.create(tokens_secret_name, BATCH_PODS_NAMESPACE, {
            'tokens.json': json.dumps({
                DEFAULT_NAMESPACE: tokens_session.session_id
            })
        })
        updates['tokens_secret_name'] = tokens_secret_name

    gsa_email = user['gsa_email']
    if gsa_email is None:
        gsa = GSAResource(iam_client)
        cleanup.append(gsa.delete)

        # length of gsa account_id must be >= 6
        assert len(ident_token) >= 6

        gsa_email, key = await gsa.create(ident_token)
        updates['gsa_email'] = gsa_email

        gsa_key_secret_name = f'{ident}-gsa-key'
        gsa_key_secret = K8sSecretResource(k8s_client)
        cleanup.append(gsa_key_secret.delete)
        await gsa_key_secret.create(gsa_key_secret_name, BATCH_PODS_NAMESPACE, {
            'key.json': base64.b64decode(key['privateKeyData']).decode('utf-8')
        })
        updates['gsa_key_secret_name'] = gsa_key_secret_name

    namespace_name = user['namespace_name']
    if namespace_name is None and user['is_developer'] == 1:
        namespace_name = ident
        namespace = K8sNamespaceResource(k8s_client)
        cleanup.append(namespace.delete)
        await namespace.create(namespace_name)
        updates['namespace_name'] = namespace_name

        db_resource = DatabaseResource(db_instance)
        cleanup.append(db_resource.delete)
        await db_resource.create(ident)

        db_secret = K8sSecretResource(k8s_client)
        cleanup.append(db_secret.delete)
        await db_secret.create(
            'database-server-config', namespace_name, db_resource.secret_data())

    n_rows = await db.execute_update(f'''
UPDATE users
SET {', '.join([f'{k} = %({k})s' for k in updates])}
WHERE id = %(id)s AND state = 'creating';
''',
                                     {'id': user['id'], **updates})
    if n_rows != 1:
        assert n_rows == 0
        raise DatabaseConflictError


async def create_user(app, user):
    cleanup = []
    try:
        await _create_user(app, user, cleanup)
    except Exception:
        log.exception(f'create user {user} failed, will retry')

        for f in cleanup:
            try:
                await f()
            except Exception:
                log.exception('caught exception while cleaning up user resource, ignoring')

        raise


async def delete_user(app, user):
    db_instance = app['db_instance']
    db = app['db']
    k8s_client = app['k8s_client']
    iam_client = app['iam_client']

    tokens_secret_name = user['tokens_secret_name']
    if tokens_secret_name is not None:
        # don't bother deleting the session since all sessions are
        # deleted below
        tokens_secret = K8sSecretResource(k8s_client, tokens_secret_name, BATCH_PODS_NAMESPACE)
        await tokens_secret.delete()

    gsa_email = user['gsa_email']
    if gsa_email is not None:
        gsa = GSAResource(iam_client, gsa_email)
        await gsa.delete()

    gsa_key_secret_name = user['gsa_key_secret_name']
    if gsa_key_secret_name is not None:
        gsa_key_secret = K8sSecretResource(k8s_client, gsa_key_secret_name, BATCH_PODS_NAMESPACE)
        await gsa_key_secret.delete()

    namespace_name = user['namespace_name']
    if namespace_name is not None:
        assert user['is_developer'] == 1

        # don't bother deleting database-server-config since we're
        # deleting the namespace
        namespace = K8sNamespaceResource(k8s_client, namespace_name)
        await namespace.delete()

        db_resource = DatabaseResource(db_instance, user['username'])
        await db_resource.delete()

    await db.just_execute('''
DELETE FROM sessions WHERE user_id = %s;
UPDATE users SET state = 'deleted' WHERE id = %s;
''',
                          (user['id'], user['id']))


async def update_users(app):
    log.info('in update_users')

    db = app['db']

    creating_users = [
        x async for x in
        db.execute_and_fetchall('SELECT * FROM users WHERE state = %s;', 'creating')]

    for user in creating_users:
        await create_user(app, user)

    deleting_users = [
        x async for x in
        db.execute_and_fetchall('SELECT * FROM users WHERE state = %s;', 'deleting')]

    for user in deleting_users:
        await delete_user(app, user)

    return True


async def async_main():
    app = {}

    try:
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

        users_changed_event = asyncio.Event()
        app['users_changed_event'] = users_changed_event

        async def users_changed_handler():
            return await update_users(app)

        user_creation_loop = EventHandler(
            users_changed_handler,
            event=users_changed_event,
            min_delay_secs=1.0)
        await user_creation_loop.start()

        while True:
            await asyncio.sleep(10000)
    finally:
        if 'db' in app:
            await app['db'].async_close()
        if 'db_instance_pool' in app:
            await app['db_instance_pool'].async_close()
