import os
import random
import json
import base64
import logging
import secrets
import asyncio
import aiohttp
import kubernetes_asyncio as kube
from hailtop.utils import time_msecs, secret_alnum_string
from hailtop.auth.sql_config import create_secret_data_from_config, SQLConfig
from hailtop import aiotools
from hailtop import batch_client as bc, httpx
from gear import create_session, Database
from gear.cloud_config import get_gcp_config, get_global_config
from gear.clients import get_identity_client

log = logging.getLogger('auth.driver')


CLOUD = get_global_config()['cloud']
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']

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
        self.task_manager = aiotools.BackgroundTaskManager()

    def shutdown(self):
        self.task_manager.shutdown()

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
                delay_secs = min(max(self.min_delay_secs, 2 * delay_secs - min(0, (ran_for_secs - t) / 2)), 30.0)

    async def bump_loop(self):
        while True:
            try:
                self.event.set()
            except Exception:
                log.exception('caught exception')
            await asyncio.sleep(self.bump_secs)

    async def start(self):
        self.task_manager.ensure_future(self.main_loop())
        if self.bump_secs is not None:
            self.task_manager.ensure_future(self.bump_loop())


class SessionResource:
    def __init__(self, db, session_id=None):
        self.db = db
        self.session_id = session_id

    async def create(self, user_id, *args, **kwargs):
        self.session_id = await create_session(self.db, user_id, *args, **kwargs)

    async def delete(self):
        if self.session_id is None:
            return

        await self.db.just_execute(
            '''
DELETE FROM sessions
WHERE session_id = %s;
''',
            (self.session_id,),
        )
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
                metadata=kube.client.V1ObjectMeta(name=name),
                data={k: base64.b64encode(v.encode('utf-8')).decode('utf-8') for k, v in data.items()},
            ),
        )
        self.name = name
        self.namespace = namespace

    async def _delete(self, name, namespace):
        try:
            await self.k8s_client.delete_namespaced_secret(name, namespace)
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

        project = get_gcp_config().project
        gsa_email = f'{username}@{project}.iam.gserviceaccount.com'

        await self._delete(gsa_email)

        service_account = await self.iam_client.post(
            '/serviceAccounts', json={"accountId": username, "serviceAccount": {"displayName": username}}
        )
        assert service_account['email'] == gsa_email
        self.gsa_email = gsa_email

        key = await self.iam_client.post(f'/serviceAccounts/{self.gsa_email}/keys')

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


class AzureServicePrincipalResource:
    def __init__(self, graph_client, app_id=None):
        self.graph_client = graph_client
        self.app_id = app_id

    async def create(self, username):
        assert self.app_id is None

        params = {'$filter': f"displayName eq '{username}'"}
        applications = await self.graph_client.get('/applications', params=params)
        for application in applications['value']:
            await self._delete(application['appId'])

        config = {'displayName': username, 'signInAudience': 'AzureADMyOrg'}
        application = await self.graph_client.post('/applications', json=config)

        self.app_id = application['appId']

        config = {'appId': application['appId']}
        service_principal = await self.graph_client.post('/servicePrincipals', json=config)

        assert self.app_id == service_principal['appId']

        password = await self.graph_client.post(f'/applications/{application["id"]}/addPassword', json={})

        credentials = {
            'appId': service_principal['appId'],
            'displayName': service_principal['displayName'],
            'name': service_principal['servicePrincipalNames'][0],
            'password': password['secretText'],
            'tenant': service_principal['appOwnerOrganizationId'],
            'objectId': service_principal['id'],
        }

        return (self.app_id, credentials)

    async def _delete(self, app_id):
        try:
            await self.graph_client.delete(f'/applications/{app_id}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                pass
            else:
                raise

    async def delete(self):
        if self.app_id is None:
            return
        await self._delete(self.app_id)
        self.app_id = None


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
        await self.db_instance.just_execute(
            f'''
CREATE DATABASE `{name}`;

CREATE USER '{name}'@'%' IDENTIFIED BY '{self.password}';
GRANT ALL ON `{name}`.* TO '{name}'@'%';
'''
        )
        self.name = name

    def secret_data(self):
        with open('/database-server-config/sql-config.json', 'r') as f:
            server_config = SQLConfig.from_json(f.read())
        with open('/database-server-config/server-ca.pem', 'r') as f:
            server_ca = f.read()
        with open('/database-server-config/client-cert.pem', 'r') as f:
            client_cert = f.read()
        with open('/database-server-config/client-key.pem', 'r') as f:
            client_key = f.read()

        if is_test_deployment:
            return create_secret_data_from_config(server_config, server_ca, client_cert, client_key)

        assert self.name is not None
        assert self.password is not None

        config = SQLConfig(
            host=server_config.host,
            port=server_config.port,
            user=self.name if CLOUD != 'azure' else f'{self.name}@{server_config.instance}',
            password=self.password,
            instance=server_config.instance,
            connection_name=server_config.connection_name,
            db=self.name,
            ssl_ca='/sql-config/server-ca.pem',
            ssl_cert='/sql-config/client-cert.pem',
            ssl_key='/sql-config/client-key.pem',
            ssl_mode='VERIFY_CA',
        )
        return create_secret_data_from_config(config, server_ca, client_cert, client_key)

    async def _delete(self, name):
        if is_test_deployment:
            return

        # no DROP USER IF EXISTS in current db version
        row = await self.db_instance.execute_and_fetchone('SELECT 1 FROM mysql.user WHERE User = %s;', (name,))
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

        await self.k8s_client.create_namespace(kube.client.V1Namespace(metadata=kube.client.V1ObjectMeta(name=name)))
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


class BillingProjectResource:
    def __init__(self, batch_client, user=None, billing_project=None):
        self.batch_client = batch_client
        self.user = user
        self.billing_project = billing_project

    async def create(self, user, billing_project):
        assert self.user is None
        assert self.billing_project is None

        await self._delete(user, billing_project)

        try:
            await self.batch_client.create_billing_project(billing_project)
        except aiohttp.ClientResponseError as e:
            if e.status != 403 or 'already exists' not in e.message:
                raise

        try:
            await self.batch_client.reopen_billing_project(billing_project)
        except aiohttp.ClientResponseError as e:
            if e.status != 403 or 'is already open' not in e.message:
                raise

        try:
            await self.batch_client.add_user(user, billing_project)
        except aiohttp.ClientResponseError as e:
            if e.status != 403 or 'already member of billing project' not in e.message:
                raise

        await self.batch_client.edit_billing_limit(billing_project, 10)

        self.user = user
        self.billing_project = billing_project

    async def _delete(self, user, billing_project):
        try:
            bp = await self.batch_client.get_billing_project(billing_project)
        except aiohttp.ClientResponseError as e:
            if e.status == 403 and 'unknown billing project':
                return
            raise
        else:
            if bp['status'] == 'closed':
                await self.batch_client.reopen_billing_project(billing_project)

        try:
            await self.batch_client.remove_user(user, billing_project)
        except aiohttp.ClientResponseError as e:
            if e.status != 403 or 'is not in billing project' not in e.message:
                raise
        finally:
            await self.batch_client.close_billing_project(billing_project)

    async def delete(self):
        if self.user is None or self.billing_project is None:
            return
        await self._delete(self.user, self.billing_project)
        self.user = None
        self.billing_project = None


async def _create_user(app, user, skip_trial_bp, cleanup):
    db_instance = app['db_instance']
    db = app['db']
    k8s_client = app['k8s_client']
    identity_client = app['identity_client']

    username = user['username']
    if user['is_service_account'] != 1:
        token = secret_alnum_string(5, case='lower')
        ident_token = f'{username}-{token}'
    else:
        token = secret_alnum_string(3, case='numbers')
        ident_token = f'{username}-{token}'

    if user['is_developer'] == 1 or user['is_service_account'] == 1 or username == 'test':
        ident = username
    else:
        ident = ident_token

    updates = {'state': 'active'}

    tokens_secret_name = user['tokens_secret_name']
    if tokens_secret_name is None:
        tokens_session = SessionResource(db)
        cleanup.append(tokens_session.delete)
        await tokens_session.create(user['id'], max_age_secs=None)

        tokens_secret_name = f'{ident}-tokens'
        tokens_secret = K8sSecretResource(k8s_client)
        cleanup.append(tokens_secret.delete)
        await tokens_secret.create(
            tokens_secret_name,
            DEFAULT_NAMESPACE,
            {'tokens.json': json.dumps({DEFAULT_NAMESPACE: tokens_session.session_id})},
        )
        updates['tokens_secret_name'] = tokens_secret_name

    hail_identity = user['hail_identity']
    if hail_identity is None:
        if CLOUD == 'gcp':
            gsa = GSAResource(identity_client)
            cleanup.append(gsa.delete)

            # length of gsa account_id must be >= 6
            assert len(ident_token) >= 6

            gsa_email, key = await gsa.create(ident_token)
            secret_data = base64.b64decode(key['privateKeyData']).decode('utf-8')
            updates['hail_identity'] = gsa_email
        else:
            assert CLOUD == 'azure'

            azure_sp = AzureServicePrincipalResource(identity_client)
            cleanup.append(azure_sp.delete)

            azure_app_id, credentials = await azure_sp.create(ident_token)
            secret_data = json.dumps(credentials)
            updates['hail_identity'] = azure_app_id

        hail_credentials_secret_name = f'{ident}-gsa-key'
        hail_identity_secret = K8sSecretResource(k8s_client)
        cleanup.append(hail_identity_secret.delete)
        await hail_identity_secret.create(
            hail_credentials_secret_name,
            DEFAULT_NAMESPACE,
            {'key.json': secret_data},
        )
        updates['hail_credentials_secret_name'] = hail_credentials_secret_name

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
        await db_secret.create('database-server-config', namespace_name, db_resource.secret_data())

    # TODO This doesn't work on azure yet because of the circular batch dependency
    # Once this is working, bootstrap must deploy batch not just auth
    if CLOUD == 'gcp' and not skip_trial_bp and user['is_service_account'] != 1:
        trial_bp = user['trial_bp_name']
        if trial_bp is None:
            batch_client = app['batch_client']
            billing_project_name = f'{username}-trial'
            billing_project = BillingProjectResource(batch_client)
            cleanup.append(billing_project.delete)
            await billing_project.create(username, billing_project_name)
            updates['trial_bp_name'] = billing_project_name

    n_rows = await db.execute_update(
        f'''
UPDATE users
SET {', '.join([f'{k} = %({k})s' for k in updates])}
WHERE id = %(id)s AND state = 'creating';
''',
        {'id': user['id'], **updates},
    )
    if n_rows != 1:
        assert n_rows == 0
        raise DatabaseConflictError


async def create_user(app, user, skip_trial_bp=False):
    cleanup = []
    try:
        await _create_user(app, user, skip_trial_bp, cleanup)
    except Exception:
        log.exception(f'create user {user} failed')

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
    identity_client = app['identity_client']

    tokens_secret_name = user['tokens_secret_name']
    if tokens_secret_name is not None:
        # don't bother deleting the session since all sessions are
        # deleted below
        tokens_secret = K8sSecretResource(k8s_client, tokens_secret_name, DEFAULT_NAMESPACE)
        await tokens_secret.delete()

    hail_identity = user['hail_identity']
    if hail_identity is not None:
        if CLOUD == 'gcp':
            gsa = GSAResource(identity_client, hail_identity)
            await gsa.delete()
        else:
            assert CLOUD == 'azure'
            azure_sp = AzureServicePrincipalResource(identity_client, hail_identity)
            await azure_sp.delete()

    hail_credentials_secret_name = user['hail_credentials_secret_name']
    if hail_credentials_secret_name is not None:
        hail_identity_secret = K8sSecretResource(k8s_client, hail_credentials_secret_name, DEFAULT_NAMESPACE)
        await hail_identity_secret.delete()

    namespace_name = user['namespace_name']
    if namespace_name is not None:
        assert user['is_developer'] == 1

        # don't bother deleting database-server-config since we're
        # deleting the namespace
        namespace = K8sNamespaceResource(k8s_client, namespace_name)
        await namespace.delete()

        db_resource = DatabaseResource(db_instance, user['username'])
        await db_resource.delete()

    trial_bp_name = user['trial_bp_name']
    if trial_bp_name is not None:
        batch_client = app['batch_client']
        bp = BillingProjectResource(batch_client, user['username'], trial_bp_name)
        await bp.delete()

    await db.just_execute(
        '''
DELETE FROM sessions WHERE user_id = %s;
UPDATE users SET state = 'deleted' WHERE id = %s;
''',
        (user['id'], user['id']),
    )


async def update_users(app):
    log.info('in update_users')

    db = app['db']

    creating_users = [x async for x in db.execute_and_fetchall('SELECT * FROM users WHERE state = %s;', 'creating')]

    for user in creating_users:
        await create_user(app, user)

    deleting_users = [x async for x in db.execute_and_fetchall('SELECT * FROM users WHERE state = %s;', 'deleting')]

    for user in deleting_users:
        await delete_user(app, user)

    return True


async def async_main():
    app = {}

    user_creation_loop = None
    try:
        db = Database()
        await db.async_init(maxsize=50)
        app['db'] = db

        app['client_session'] = httpx.client_session()

        db_instance = Database()
        await db_instance.async_init(maxsize=50, config_file='/database-server-config/sql-config.json')
        app['db_instance'] = db_instance

        kube.config.load_incluster_config()
        k8s_client = kube.client.CoreV1Api()
        app['k8s_client'] = k8s_client

        app['identity_client'] = get_identity_client()

        app['batch_client'] = await bc.aioclient.BatchClient.create(None)

        users_changed_event = asyncio.Event()
        app['users_changed_event'] = users_changed_event

        async def users_changed_handler():
            return await update_users(app)

        user_creation_loop = EventHandler(users_changed_handler, event=users_changed_event, min_delay_secs=1.0)
        await user_creation_loop.start()

        while True:
            await asyncio.sleep(10000)
    finally:
        try:
            if 'db' in app:
                await app['db'].async_close()
        finally:
            try:
                if 'db_instance_pool' in app:
                    await app['db_instance_pool'].async_close()
            finally:
                try:
                    await app['client_session'].close()
                finally:
                    try:
                        if user_creation_loop is not None:
                            user_creation_loop.shutdown()
                    finally:
                        await app['identity_client'].close()
