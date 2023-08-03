import asyncio
import logging
import os
import secrets
from functools import wraps

import aiohttp
import aiohttp_session
import aiohttp_session.cookie_storage
import kubernetes_asyncio.client
import kubernetes_asyncio.client.rest
import kubernetes_asyncio.config
import pymysql
from aiohttp import web
from prometheus_async.aio.web import server_stats  # type: ignore

from gear import AuthClient, check_csrf_token, create_database_pool, monitor_endpoints_middleware, setup_aiohttp_session
from gear.cloud_config import get_global_config
from hailtop import httpx
from hailtop.config import get_deploy_config
from hailtop.hail_logging import AccessLogger
from hailtop.tls import internal_server_ssl_context
from web_common import render_template, sass_compile, set_message, setup_aiohttp_jinja2, setup_common_static_routes

log = logging.getLogger('notebook')

NOTEBOOK_NAMESPACE = os.environ['HAIL_NOTEBOOK_NAMESPACE']

deploy_config = get_deploy_config()

routes = web.RouteTableDef()

auth = AuthClient()

# Must be int for Kubernetes V1 api timeout_seconds property
KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5))

POD_PORT = 8888

DEFAULT_WORKER_IMAGE = os.environ['HAIL_NOTEBOOK_WORKER_IMAGE']

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')


async def workshop_userdata_from_web_request(request):
    session = await aiohttp_session.get_session(request)
    if 'workshop_session' not in session:
        return None
    workshop_session = session['workshop_session']

    # verify this workshop is active
    name = workshop_session['workshop_name']
    token = workshop_session['workshop_token']

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                'SELECT * FROM workshops WHERE name = %s AND token = %s AND active = 1;', (name, token)
            )
            workshops = await cursor.fetchall()

            if len(workshops) != 1:
                assert len(workshops) == 0
                del session['workshop_session']
                return None
            workshop = workshops[0]

    return {'id': workshop_session['id'], 'workshop': workshop}


def web_maybe_authenticated_workshop_guest(fun):
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        return await fun(request, await workshop_userdata_from_web_request(request), *args, **kwargs)

    return wrapped


def web_authenticated_workshop_guest_only(redirect=True):
    def wrap(fun):
        @web_maybe_authenticated_workshop_guest
        @wraps(fun)
        async def wrapped(request, userdata, *args, **kwargs):
            if not userdata:
                if redirect:
                    raise web.HTTPFound(deploy_config.external_url('workshop', '/login'))
                raise web.HTTPUnauthorized()
            return await fun(request, userdata, *args, **kwargs)

        return wrapped

    return wrap


async def start_pod(k8s, service, userdata, notebook_token, jupyter_token):
    service_base_path = deploy_config.base_path(service)

    origin = deploy_config.external_url('workshop', '/').rstrip('/')

    command = [
        'jupyter',
        'notebook',
        f'--NotebookApp.token={jupyter_token}',
        f'--NotebookApp.base_url={service_base_path}/instance/{notebook_token}/',
        "--ip",
        "0.0.0.0",
        f"--NotebookApp.allow_origin={origin}",
        "--no-browser",
        "--allow-root",
    ]

    if service == 'notebook':
        service_account_name = userdata.get('ksa_name')

        bucket = userdata['bucket_name']
        command.append(f'--GoogleStorageContentManager.default_path="{bucket}"')

        image = DEFAULT_WORKER_IMAGE

        env = [
            kubernetes_asyncio.client.V1EnvVar(
                name='HAIL_DEPLOY_CONFIG_FILE', value='/deploy-config/deploy-config.json'
            )
        ]

        tokens_secret_name = userdata['tokens_secret_name']
        hail_credentials_secret_name = userdata['hail_credentials_secret_name']
        volumes = [
            kubernetes_asyncio.client.V1Volume(
                name='deploy-config', secret=kubernetes_asyncio.client.V1SecretVolumeSource(secret_name='deploy-config')
            ),
            kubernetes_asyncio.client.V1Volume(
                name='gsa-key',
                secret=kubernetes_asyncio.client.V1SecretVolumeSource(secret_name=hail_credentials_secret_name),
            ),
            kubernetes_asyncio.client.V1Volume(
                name='user-tokens',
                secret=kubernetes_asyncio.client.V1SecretVolumeSource(secret_name=tokens_secret_name),
            ),
        ]
        volume_mounts = [
            kubernetes_asyncio.client.V1VolumeMount(mount_path='/deploy-config', name='deploy-config', read_only=True),
            kubernetes_asyncio.client.V1VolumeMount(mount_path='/gsa-key', name='gsa-key', read_only=True),
            kubernetes_asyncio.client.V1VolumeMount(mount_path='/user-tokens', name='user-tokens', read_only=True),
        ]
        resources = kubernetes_asyncio.client.V1ResourceRequirements(requests={'cpu': '1.601', 'memory': '1.601G'})
    else:
        workshop = userdata['workshop']

        service_account_name = None
        image = workshop['image']
        env = []
        volumes = []
        volume_mounts = []

        cpu = workshop['cpu']
        memory = workshop['memory']
        resources = kubernetes_asyncio.client.V1ResourceRequirements(
            requests={'cpu': cpu, 'memory': memory}, limits={'cpu': cpu, 'memory': memory}
        )

    pod_spec = kubernetes_asyncio.client.V1PodSpec(
        node_selector={'preemptible': 'false'},
        service_account_name=service_account_name,
        containers=[
            kubernetes_asyncio.client.V1Container(
                command=command,
                name='default',
                image=image,
                env=env,
                ports=[kubernetes_asyncio.client.V1ContainerPort(container_port=POD_PORT)],
                resources=resources,
                volume_mounts=volume_mounts,
            )
        ],
        volumes=volumes,
    )

    user_id = str(userdata['id'])
    pod_template = kubernetes_asyncio.client.V1Pod(
        metadata=kubernetes_asyncio.client.V1ObjectMeta(
            generate_name='notebook-worker-', labels={'app': 'notebook-worker', 'user_id': user_id}
        ),
        spec=pod_spec,
    )
    pod = await k8s.create_namespaced_pod(
        NOTEBOOK_NAMESPACE, pod_template, _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS
    )

    return pod


def notebook_status_from_pod(pod):
    pod_ip = pod.status.pod_ip
    if not pod_ip:
        state = 'Scheduling'
    else:
        state = 'Initializing'
        if pod.status and pod.status.conditions:
            for c in pod.status.conditions:
                if c.type == 'Ready' and c.status == 'True':
                    state = 'Initializing'
    return {'pod_ip': pod_ip, 'state': state}


async def k8s_notebook_status_from_notebook(k8s, notebook):
    if not notebook:
        return None

    try:
        pod = await k8s.read_namespaced_pod(
            name=notebook['pod_name'], namespace=NOTEBOOK_NAMESPACE, _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS
        )
        return notebook_status_from_pod(pod)
    except kubernetes_asyncio.client.rest.ApiException as e:
        if e.status == 404:
            log.exception(f"404 for pod: {notebook['pod_name']}")
            return None
        raise


async def notebook_status_from_notebook(client_session: httpx.ClientSession, k8s, service, headers, cookies, notebook):
    status = await k8s_notebook_status_from_notebook(k8s, notebook)
    if not status:
        return None

    if status['state'] == 'Initializing':
        if notebook['state'] == 'Ready':
            status['state'] = 'Ready'
        else:
            pod_name = notebook['pod_name']

            # don't have dev credentials to connect through internal.hail.is
            ready_url = deploy_config.external_url(
                service, f'/instance/{notebook["notebook_token"]}/?token={notebook["jupyter_token"]}'
            )
            try:
                async with client_session.get(ready_url, headers=headers, cookies=cookies) as resp:
                    if resp.status >= 200 and resp.status < 300:
                        log.info(f'GET on jupyter pod {pod_name} succeeded: {resp}')
                        status['state'] = 'Ready'
                    else:
                        log.info(f'GET on jupyter pod {pod_name} failed: {resp}')
            except aiohttp.ServerTimeoutError:
                log.exception(f'GET on jupyter pod {pod_name} timed out: {resp}')

    return status


async def update_notebook_return_changed(dbpool, user_id, notebook, new_status):
    if not new_status:
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('DELETE FROM notebooks WHERE user_id = %s;', user_id)
                return True
    if new_status['state'] != notebook['state']:
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    'UPDATE notebooks SET state = %s, pod_ip = %s WHERE user_id = %s;',
                    (new_status['state'], new_status['pod_ip'], user_id),
                )
        return True
    return False


async def get_user_notebook(dbpool, user_id):
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM notebooks WHERE user_id = %s;', user_id)
            notebooks = await cursor.fetchall()

    if len(notebooks) == 1:
        return notebooks[0]
    assert len(notebooks) == 0, len(notebooks)
    return None


async def delete_worker_pod(k8s, pod_name):
    try:
        await k8s.delete_namespaced_pod(pod_name, NOTEBOOK_NAMESPACE, _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
    except kubernetes_asyncio.client.rest.ApiException as e:
        log.info(f'pod {pod_name} already deleted {e}')


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('')
@routes.get('/')
@auth.web_maybe_authenticated_user
async def index(request, userdata):  # pylint: disable=unused-argument
    return await render_template('notebook', request, userdata, 'index.html', {})


async def _get_notebook(service, request, userdata):
    app = request.app
    dbpool = app['dbpool']
    page_context = {'notebook': await get_user_notebook(dbpool, str(userdata['id'])), 'notebook_service': service}
    return await render_template(service, request, userdata, 'notebook.html', page_context)


async def _post_notebook(service, request, userdata):
    app = request.app
    dbpool = app['dbpool']
    k8s = app['k8s_client']

    notebook_token = secrets.token_urlsafe(32)
    jupyter_token = secrets.token_hex(16)

    pod = await start_pod(k8s, service, userdata, notebook_token, jupyter_token)
    if pod.status.pod_ip:
        state = 'Initializing'
    else:
        state = 'Scheduling'

    user_id = str(userdata['id'])
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                '''
DELETE FROM notebooks WHERE user_id = %s;
INSERT INTO notebooks (user_id, notebook_token, pod_name, state, pod_ip, jupyter_token) VALUES (%s, %s, %s, %s, %s, %s);
''',
                (user_id, user_id, notebook_token, pod.metadata.name, state, pod.status.pod_ip, jupyter_token),
            )

    raise web.HTTPFound(location=deploy_config.external_url(service, '/notebook'))


async def _delete_notebook(service, request, userdata):
    app = request.app
    dbpool = app['dbpool']
    k8s = app['k8s_client']
    user_id = str(userdata['id'])
    notebook = await get_user_notebook(dbpool, user_id)
    if notebook:
        await delete_worker_pod(k8s, notebook['pod_name'])
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('DELETE FROM notebooks WHERE user_id = %s;', user_id)

    raise web.HTTPFound(location=deploy_config.external_url(service, '/notebook'))


async def _wait_websocket(service, request, userdata):
    app = request.app
    k8s = app['k8s_client']
    dbpool = app['dbpool']
    client_session: httpx.ClientSession = app['client_session']
    user_id = str(userdata['id'])
    notebook = await get_user_notebook(dbpool, user_id)
    if not notebook:
        return web.HTTPNotFound()

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # forward authorization
    headers = {}
    if 'Authorization' in request.headers:
        headers['Authorization'] = request.headers['Authorization']
    if 'X-Hail-Internal-Authorization' in request.headers:
        headers['X-Hail-Internal-Authorization'] = request.headers['X-Hail-Internal-Authorization']

    cookies = {}
    cloud = get_global_config()['cloud']
    for k in (f'{cloud}_session', f'{cloud}_sesh'):
        if k in request.cookies:
            cookies[k] = request.cookies[k]

    ready = notebook['state'] == 'Ready'
    count = 0
    while count < 10:
        try:
            new_status = await notebook_status_from_notebook(client_session, k8s, service, headers, cookies, notebook)
            changed = await update_notebook_return_changed(dbpool, user_id, notebook, new_status)
            if changed:
                log.info(f"pod {notebook['pod_name']} status changed: {notebook['state']} => {new_status['state']}")
                break
        except Exception:  # pylint: disable=broad-except
            log.exception(f"/wait: error while updating status for pod: {notebook['pod_name']}")
        await asyncio.sleep(1)
        count += 1

    ready = new_status and new_status['state'] == 'Ready'

    # 0/1 ready
    await ws.send_str(str(int(ready)))

    return ws


async def _get_error(service, request, userdata):
    if not userdata:
        raise web.HTTPFound(deploy_config.external_url(service, '/login'))

    app = request.app
    k8s = app['k8s_client']
    dbpool = app['dbpool']
    user_id = str(userdata['id'])

    # we just failed a check, so update status from k8s without probe,
    # best we can do is 'Initializing'
    notebook = await get_user_notebook(dbpool, user_id)
    new_status = await k8s_notebook_status_from_notebook(k8s, notebook)
    await update_notebook_return_changed(dbpool, user_id, notebook, new_status)

    session = await aiohttp_session.get_session(request)
    if notebook:
        if new_status['state'] == 'Ready':
            raise web.HTTPFound(
                deploy_config.external_url(
                    service, f'/instance/{notebook["notebook_token"]}/?token={notebook["jupyter_token"]}'
                )
            )
        set_message(
            session,
            'Could not connect to Jupyter instance.  Please wait for Jupyter to be ready and try again.',
            'error',
        )
    else:
        set_message(session, 'Jupyter instance not found.  Please launch a new instance.', 'error')
    raise web.HTTPFound(deploy_config.external_url(service, '/notebook'))


async def _get_auth(request, userdata):
    requested_notebook_token = request.match_info['requested_notebook_token']
    app = request.app
    dbpool = app['dbpool']

    notebook = await get_user_notebook(dbpool, str(userdata['id']))
    if notebook and notebook['notebook_token'] == requested_notebook_token:
        pod_ip = notebook['pod_ip']
        if pod_ip:
            return web.Response(headers={'pod_ip': f'{pod_ip}:{POD_PORT}'})

    return web.HTTPForbidden()


@routes.get('/notebook')
@auth.web_authenticated_users_only()
async def get_notebook(request, userdata):
    return await _get_notebook('notebook', request, userdata)


@routes.post('/notebook/delete')
@check_csrf_token
@auth.web_authenticated_users_only(redirect=False)
async def delete_notebook(request, userdata):  # pylint: disable=unused-argument
    return await _delete_notebook('notebook', request, userdata)


@routes.post('/notebook')
@check_csrf_token
@auth.web_authenticated_users_only(redirect=False)
async def post_notebook(request, userdata):
    return await _post_notebook('notebook', request, userdata)


@routes.get('/auth/{requested_notebook_token}')
@auth.web_authenticated_users_only(redirect=False)
async def get_auth(request, userdata):
    return await _get_auth(request, userdata)


@routes.get('/images')
async def get_images(request):
    images = [DEFAULT_WORKER_IMAGE]

    app = request.app
    dbpool = app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT image FROM workshops WHERE active = 1;')
            workshops = await cursor.fetchall()
    for workshop in workshops:
        images.append(workshop['image'])

    return web.Response(text=' '.join(images))


@routes.get('/notebook/wait')
@auth.web_authenticated_users_only(redirect=False)
async def wait_websocket(request, userdata):
    return await _wait_websocket('notebook', request, userdata)


@routes.get('/error')
@auth.web_maybe_authenticated_user
async def get_error(request, userdata):
    return await _get_error('notebook', request, userdata)


@routes.get('/workshop-admin')
@auth.web_authenticated_developers_only()
async def workshop_admin(request, userdata):
    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM workshops')
            workshops = await cursor.fetchall()

    page_context = {'workshops': workshops}
    return await render_template('notebook', request, userdata, 'workshop-admin.html', page_context)


@routes.post('/workshop-admin-create')
@check_csrf_token
@auth.web_authenticated_developers_only()
async def create_workshop(request, userdata):  # pylint: disable=unused-argument
    dbpool = request.app['dbpool']
    session = await aiohttp_session.get_session(request)

    post = await request.post()
    name = post['name']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                active = post.get('active') == 'on'
                if active:
                    token = secrets.token_urlsafe(32)
                else:
                    token = None
                await cursor.execute(
                    '''
INSERT INTO workshops (name, image, cpu, memory, password, active, token) VALUES (%s, %s, %s, %s, %s, %s, %s);
''',
                    (name, post['image'], post['cpu'], post['memory'], post['password'], active, token),
                )
                set_message(session, f'Created workshop {name}.', 'info')
            except pymysql.err.IntegrityError as e:
                if e.args[0] == 1062:  # duplicate error
                    set_message(session, f'Cannot create workshop {name}: duplicate name.', 'error')
                else:
                    raise

    raise web.HTTPFound(deploy_config.external_url('notebook', '/workshop-admin'))


@routes.post('/workshop-admin-update')
@check_csrf_token
@auth.web_authenticated_developers_only()
async def update_workshop(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']

    post = await request.post()
    name = post['name']
    id = post['id']
    session = await aiohttp_session.get_session(request)
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            active = post.get('active') == 'on'
            # FIXME don't set token unless re-activating
            if active:
                token = secrets.token_urlsafe(32)
            else:
                token = None
            n = await cursor.execute(
                '''
UPDATE workshops SET name = %s, image = %s, cpu = %s, memory = %s, password = %s, active = %s, token = %s WHERE id = %s;
''',
                (name, post['image'], post['cpu'], post['memory'], post['password'], active, token, id),
            )
            if n == 0:
                set_message(session, f'Internal error: cannot update workshop: workshop ID {id} not found.', 'error')
            else:
                set_message(session, f'Updated workshop {name}.', 'info')

    raise web.HTTPFound(deploy_config.external_url('notebook', '/workshop-admin'))


@routes.post('/workshop-admin-delete')
@check_csrf_token
@auth.web_authenticated_developers_only()
async def delete_workshop(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']

    post = await request.post()
    name = post['name']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            n = await cursor.execute(
                '''
DELETE FROM workshops WHERE name = %s;
''',
                name,
            )

    session = await aiohttp_session.get_session(request)
    if n == 1:
        set_message(session, f'Deleted workshop {name}.', 'info')
    else:
        set_message(session, f'Workshop {name} not found.', 'error')

    raise web.HTTPFound(deploy_config.external_url('notebook', '/workshop-admin'))


workshop_routes = web.RouteTableDef()


@workshop_routes.get('')
@workshop_routes.get('/')
@web_maybe_authenticated_workshop_guest
async def workshop_get_index(request, userdata):
    page_context = {'notebook_service': 'workshop'}
    return await render_template('workshop', request, userdata, 'workshop/index.html', page_context)


@workshop_routes.get('/login')
@web_maybe_authenticated_workshop_guest
async def workshop_get_login(request, userdata):
    if userdata:
        raise web.HTTPFound(location=deploy_config.external_url('workshop', '/notebook'))

    page_context = {'notebook_service': 'workshop'}
    return await render_template('workshop', request, userdata, 'workshop/login.html', page_context)


@workshop_routes.post('/login')
@check_csrf_token
async def workshop_post_login(request):
    session = await aiohttp_session.get_session(request)
    dbpool = request.app['dbpool']

    post = await request.post()
    name = post['name']
    password = post['password']

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                '''
SELECT * FROM workshops
WHERE name = %s AND password = %s AND active = 1;
''',
                (name, password),
            )
            workshops = await cursor.fetchall()

            if len(workshops) != 1:
                assert len(workshops) == 0
                set_message(session, 'Workshop Inactive!', 'error')
                raise web.HTTPFound(location=deploy_config.external_url('workshop', '/login'))
            workshop = workshops[0]

    # use hex since K8s labels can't start or end with _ or -
    user_id = secrets.token_hex(16)
    session['workshop_session'] = {'workshop_name': name, 'workshop_token': workshop['token'], 'id': user_id}

    set_message(session, f'Welcome to the {name} workshop!', 'info')

    raise web.HTTPFound(location=deploy_config.external_url('workshop', '/notebook'))


@workshop_routes.post('/logout')
@check_csrf_token
@web_authenticated_workshop_guest_only(redirect=True)
async def workshop_post_logout(request, userdata):
    app = request.app
    dbpool = app['dbpool']
    k8s = app['k8s_client']
    user_id = str(userdata['id'])
    notebook = await get_user_notebook(dbpool, user_id)
    if notebook:
        # Notebook is inaccessible since login creates a new random
        # user id, so delete it.
        await delete_worker_pod(k8s, notebook['pod_name'])
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('DELETE FROM notebooks WHERE user_id = %s;', user_id)

    session = await aiohttp_session.get_session(request)
    if 'workshop_session' in session:
        del session['workshop_session']

    raise web.HTTPFound(location=deploy_config.external_url('workshop', '/notebook'))


@workshop_routes.get('/resources')
@web_maybe_authenticated_workshop_guest
async def workshop_get_faq(request, userdata):
    page_context = {'notebook_service': 'workshop'}
    return await render_template('workshop', request, userdata, 'workshop/resources.html', page_context)


@workshop_routes.get('/notebook')
@web_authenticated_workshop_guest_only()
async def workshop_get_notebook(request, userdata):
    return await _get_notebook('workshop', request, userdata)


@workshop_routes.post('/notebook')
@check_csrf_token
@web_authenticated_workshop_guest_only(redirect=False)
async def workshop_post_notebook(request, userdata):
    return await _post_notebook('workshop', request, userdata)


@workshop_routes.get('/auth/{requested_notebook_token}')
@web_authenticated_workshop_guest_only(redirect=False)
async def workshop_get_auth(request, userdata):
    return await _get_auth(request, userdata)


@workshop_routes.post('/notebook/delete')
@check_csrf_token
@web_authenticated_workshop_guest_only(redirect=False)
async def workshop_delete_notebook(request, userdata):
    return await _delete_notebook('workshop', request, userdata)


@workshop_routes.get('/notebook/wait')
@web_authenticated_workshop_guest_only(redirect=False)
async def workshop_wait_websocket(request, userdata):
    return await _wait_websocket('workshop', request, userdata)


@workshop_routes.get('/error')
@auth.web_maybe_authenticated_user
async def workshop_get_error(request, userdata):
    return await _get_error('workshop', request, userdata)


async def on_startup(app):
    if 'BATCH_USE_KUBE_CONFIG' in os.environ:
        await kubernetes_asyncio.config.load_kube_config()
    else:
        kubernetes_asyncio.config.load_incluster_config()
    app['k8s_client'] = kubernetes_asyncio.client.CoreV1Api()

    app['dbpool'] = await create_database_pool()

    app['client_session'] = httpx.client_session()


async def on_cleanup(app):
    try:
        del app['k8s_client']
    finally:
        try:
            await app['client_session'].close()
        finally:
            await asyncio.gather(*(t for t in asyncio.all_tasks() if t is not asyncio.current_task()))


def init_app(routes):
    app = web.Application(middlewares=[monitor_endpoints_middleware])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    setup_aiohttp_jinja2(app, 'notebook')
    setup_aiohttp_session(app)

    root = os.path.dirname(os.path.abspath(__file__))
    routes.static('/static', f'{root}/static')
    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.router.add_get("/metrics", server_stats)

    return app


def run():
    sass_compile('notebook')

    notebook_app = init_app(routes)
    workshop_app = init_app(workshop_routes)

    root_app = web.Application()
    root_app.add_domain('notebook*', deploy_config.prefix_application(notebook_app, 'notebook'))
    root_app.add_domain('workshop*', deploy_config.prefix_application(workshop_app, 'workshop'))
    root_app.router.add_get("/metrics", server_stats)
    web.run_app(
        root_app, host='0.0.0.0', port=5000, access_log_class=AccessLogger, ssl_context=internal_server_ssl_context()
    )
