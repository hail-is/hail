import logging
import os
import secrets
from functools import wraps
import asyncio
import pymysql
import aiohttp
from aiohttp import web
import aiohttp_session
import aiohttp_session.cookie_storage
from kubernetes_asyncio import client, config
import kubernetes_asyncio as kube

from hailtop.config import get_deploy_config
from gear import setup_aiohttp_session, create_database_pool, \
    web_authenticated_users_only, web_maybe_authenticated_user, web_authenticated_developers_only, \
    check_csrf_token
from web_common import sass_compile, setup_aiohttp_jinja2, setup_common_static_routes, \
    set_message, render_template

log = logging.getLogger('notebook')

NOTEBOOK_NAMESPACE = os.environ['HAIL_NOTEBOOK_NAMESPACE']

deploy_config = get_deploy_config()

routes = web.RouteTableDef()

# Must be int for Kubernetes V1 api timeout_seconds property
KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5))

POD_PORT = 8888

DEFAULT_WORKER_IMAGE = os.environ['HAIL_NOTEBOOK_WORKER_IMAGE']

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')


async def workshop_userdata_from_web_request(request):
    session = await aiohttp_session.get_session(request)
    if 'workshop_session' not in session:
        return None
    userdata = session['workshop_session']

    # verify this workshop is active
    name = userdata['workshop_name']
    token = userdata['workshop_token']

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                'SELECT * FROM workshops WHERE name = %s AND token = %s',
                (name, token))
            workshops = await cursor.fetchall()

    if len(workshops) != 1:
        return None

    return userdata


def web_maybe_authenticated_workshop_guest(fun):
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        return await fun(request, await workshop_userdata_from_web_request(request), *args, **kwargs)
    return wrapped


def web_authenticated_workshop_guest_only(fun):
    @web_maybe_authenticated_workshop_guest
    @wraps(fun)
    async def wrapped(request, userdata, *args, **kwargs):
        if not userdata:
            raise web.HTTPFound(deploy_config.external_url('workshop', '/login'))
        return await fun(request, userdata, *args, **kwargs)
    return wrapped


async def start_pod(k8s, service, userdata, notebook_token, jupyter_token):
    service_base_path = deploy_config.base_path(service)

    command = [
        'jupyter',
        'notebook',
        f'--NotebookApp.token={jupyter_token}',
        f'--NotebookApp.base_url={service_base_path}/instance/{notebook_token}/',
        "--ip", "0.0.0.0", "--no-browser", "--allow-root"
    ]
    if 'workshop_image' in userdata:
        image = userdata['workshop_image']
    else:
        image = DEFAULT_WORKER_IMAGE
    volumes = [
        kube.client.V1Volume(
            name='deploy-config',
            secret=kube.client.V1SecretVolumeSource(
                secret_name='deploy-config'))
    ]
    volume_mounts = [
        kube.client.V1VolumeMount(
            mount_path='/deploy-config',
            name='deploy-config',
            read_only=True)
    ]

    user_id = userdata['id']
    ksa_name = userdata.get('ksa_name')

    bucket = userdata.get('bucket_name')
    if bucket is not None:
        command.append(f'--GoogleStorageContentManager.default_path="{bucket}"')

    gsa_key_secret_name = userdata.get('gsa_key_secret_name')
    if gsa_key_secret_name is not None:
        volumes.append(
            kube.client.V1Volume(
                name='gsa-key',
                secret=kube.client.V1SecretVolumeSource(
                    secret_name=gsa_key_secret_name)))
        volume_mounts.append(
            kube.client.V1VolumeMount(
                mount_path='/gsa-key',
                name='gsa-key',
                read_only=True))

    jwt_secret_name = userdata.get('jwt_secret_name')
    if jwt_secret_name is not None:
        volumes.append(
            kube.client.V1Volume(
                name='user-tokens',
                secret=kube.client.V1SecretVolumeSource(
                    secret_name=jwt_secret_name)))
        volume_mounts.append(
            kube.client.V1VolumeMount(
                mount_path='/user-tokens',
                name='user-tokens',
                read_only=True))

    pod_spec = kube.client.V1PodSpec(
        service_account_name=ksa_name,
        containers=[
            kube.client.V1Container(
                command=command,
                name='default',
                image=image,
                env=[kube.client.V1EnvVar(name='HAIL_DEPLOY_CONFIG_FILE',
                                          value='/deploy-config/deploy-config.json')],
                ports=[kube.client.V1ContainerPort(container_port=POD_PORT)],
                resources=kube.client.V1ResourceRequirements(
                    requests={'cpu': '1.601', 'memory': '1.601G'}),
                volume_mounts=volume_mounts)
        ],
        volumes=volumes)

    pod_template = kube.client.V1Pod(
        metadata=kube.client.V1ObjectMeta(
            generate_name='notebook-worker-',
            labels={
                'app': 'notebook-worker',
                'user_id': str(user_id)
            }),
        spec=pod_spec)
    pod = await k8s.create_namespaced_pod(
        NOTEBOOK_NAMESPACE,
        pod_template,
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)

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
                    state = 'Running'
    return {
        'pod_ip': pod_ip,
        'state': state
    }


async def notebook_status_from_notebook(k8s, service, headers, cookies, notebook):
    try:
        pod = await k8s.read_namespaced_pod(
            name=notebook['pod_name'],
            namespace=NOTEBOOK_NAMESPACE,
            _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
    except kube.client.rest.ApiException as e:
        if e.status == 404:
            return None
        raise

    status = notebook_status_from_pod(pod)

    if status['state'] == 'Running':
        if notebook['state'] == 'Ready':
            status['state'] = 'Ready'
        else:
            pod_name = notebook['pod_name']

            # don't have dev credentials to connect through internal.hail.is
            ready_url = deploy_config.external_url(
                service,
                f'/instance/{notebook["notebook_token"]}/?token={notebook["jupyter_token"]}')
            try:
                async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=1),
                        headers=headers,
                        cookies=cookies) as session:
                    async with session.get(ready_url) as resp:
                        if resp.status >= 200 and resp.status < 300:
                            log.info(f'GET on jupyter pod {pod_name} succeeded: {resp}')
                            status['state'] = 'Ready'
                        else:
                            log.info(f'GET on jupyter pod {pod_name} failed: {resp}')
            except aiohttp.ServerTimeoutError:
                log.info(f'GET on jupyter pod {pod_name} timed out: {resp}')

    return status


async def get_user_notebook(app, user_id):
    dbpool = app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM notebooks WHERE user_id = %s;', user_id)
            notebooks = await cursor.fetchall()

    if len(notebooks) == 1:
        return notebooks[0]
    return None


async def delete_worker_pod(k8s, pod_name):
    try:
        await k8s.delete_namespaced_pod(
            pod_name,
            NOTEBOOK_NAMESPACE,
            _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
    except kube.client.rest.ApiException as e:
        log.info(f'pod {pod_name} already deleted {e}')


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response()


@routes.get('')
@routes.get('/')
@web_maybe_authenticated_user
async def index(request, userdata):  # pylint: disable=unused-argument
    return await render_template('notebook', request, userdata, 'index.html', {})


async def _get_notebook(service, request, userdata):
    page_context = {
        'notebook': await get_user_notebook(request.app, userdata['id']),
        'notebook_service': service
    }
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

    user_id = userdata['id']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                '''
DELETE FROM notebooks WHERE user_id = %s;
INSERT INTO notebooks (user_id, notebook_token, pod_name, state, pod_ip, jupyter_token) VALUES (%s, %s, %s, %s, %s, %s);
''',
                (user_id, user_id, notebook_token, pod.metadata.name, state, pod.status.pod_ip, jupyter_token))

    return web.HTTPFound(
        location=deploy_config.external_url(service, '/notebook'))


async def _delete_notebook_2(request, userdata):
    app = request.app
    dbpool = app['dbpool']
    k8s = app['k8s_client']

    notebook = await get_user_notebook(app, userdata['id'])
    if notebook:
        await delete_worker_pod(k8s, notebook['pod_name'])
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    'DELETE FROM notebooks WHERE user_id = %s;', userdata['id'])


async def _delete_notebook(service, request, userdata):
    await _delete_notebook_2(request, userdata)
    return web.HTTPFound(location=deploy_config.external_url(service, '/notebook'))


async def _wait_websocket(service, request, userdata):
    app = request.app
    notebook = await get_user_notebook(app, userdata['id'])
    if not notebook:
        return web.HTTPNotFound()

    k8s = request.app['k8s_client']
    dbpool = request.app['dbpool']

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # forward authorization
    headers = {}
    if 'Authorization' in request.headers:
        headers['Authorization'] = request.headers['Authorization']
    if 'X-Hail-Internal-Authorization' in request.headers:
        headers['X-Hail-Internal-Authorization'] = request.headers['X-Hail-Internal-Authorization']

    cookies = {}
    if 'session' in request.cookies:
        cookies['session'] = request.cookies['session']
    if 'sesh' in request.cookies:
        cookies['sesh'] = request.cookies['sesh']

    ready = (notebook['state'] == 'Ready')
    count = 0
    while count < 10:
        status = await notebook_status_from_notebook(k8s, service, headers, cookies, notebook)
        if not status:
            async with dbpool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        'DELETE FROM notebooks WHERE user_id = %s;',
                        userdata['id'])
            ready = False
            break
        if status['state'] != notebook['state']:
            async with dbpool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        'UPDATE notebooks SET state = %s, pod_ip = %s WHERE user_id = %s;',
                        (status['state'], status['pod_ip'], userdata['id']))
            ready = (status['state'] == 'Ready')
            break

        await asyncio.sleep(1)
        count += 1

    # 0/1 ready
    await ws.send_str(str(int(ready)))

    return ws


async def _get_error(service, request, userdata):
    if not userdata:
        return web.HTTPFound(deploy_config.external_url(service, '/login'))

    await _delete_notebook_2(request, userdata)

    session = await aiohttp_session.get_session(request)
    set_message(session,
                f'Notebook not found.  Please create a new notebook.',
                'error')
    return web.HTTPFound(deploy_config.external_url(service, '/notebook'))


async def _get_auth(request, userdata):
    app = request.app
    requested_notebook_token = request.match_info['requested_notebook_token']

    notebook = await get_user_notebook(app, userdata['id'])
    if notebook and notebook['notebook_token'] == requested_notebook_token:
        pod_ip = notebook['pod_ip']
        if pod_ip:
            return web.Response(headers={
                'pod_ip': f'{pod_ip}:{POD_PORT}'
            })

    return web.HTTPNotFound()


@routes.get('/notebook')
@web_authenticated_users_only()
async def get_notebook(request, userdata):
    return await _get_notebook('notebook', request, userdata)


@routes.post('/notebook/delete')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def delete_notebook(request, userdata):  # pylint: disable=unused-argument
    return await _delete_notebook('notebook', request, userdata)


@routes.post('/notebook')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def post_notebook(request, userdata):
    return await _post_notebook('notebook', request, userdata)


@routes.get('/auth/{requested_notebook_token}')
@web_authenticated_users_only()
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
@web_authenticated_users_only(redirect=False)
async def wait_websocket(request, userdata):
    return await _wait_websocket('notebook', request, userdata)


@routes.get('/error')
@web_maybe_authenticated_user
async def get_error(request, userdata):
    return await _get_error('notebook', request, userdata)


@routes.get('/user')
@web_authenticated_users_only()
async def user_page(request, userdata):
    return await render_template('notebook', request, userdata, 'user.html', {})


@routes.get('/workshop-admin')
@web_authenticated_developers_only()
async def workshop_admin(request, userdata):
    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM workshops')
            workshops = await cursor.fetchall()

    page_context = {
        'workshops': workshops
    }
    return await render_template('notebook', request, userdata, 'workshop-admin.html', page_context)


@routes.post('/workshop-admin-create')
@check_csrf_token
@web_authenticated_developers_only()
async def create_workshop(request, userdata):  # pylint: disable=unused-argument
    dbpool = request.app['dbpool']
    session = await aiohttp_session.get_session(request)

    post = await request.post()
    name = post['name']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                active = (post.get('active') == 'on')
                if active:
                    token = secrets.token_urlsafe(32)
                else:
                    token = None
                await cursor.execute('''
INSERT INTO workshops (name, image, password, active, token) VALUES (%s, %s, %s, %s, %s);
''',
                                     (name,
                                      post['image'],
                                      post['password'],
                                      active,
                                      token))
                set_message(session, f'Created workshop {name}.', 'info')
            except pymysql.err.IntegrityError as e:
                if e.args[0] == 1062:  # duplicate error
                    set_message(session,
                                f'Cannot create workshop {name}: duplicate name.',
                                'error')
                else:
                    raise

    return web.HTTPFound(deploy_config.external_url('notebook', '/workshop-admin'))


@routes.post('/workshop-admin-update')
@check_csrf_token
@web_authenticated_developers_only()
async def update_workshop(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']

    post = await request.post()
    name = post['name']
    id = post['id']
    session = await aiohttp_session.get_session(request)
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            active = (post.get('active') == 'on')
            # FIXME don't set token unless re-activating
            if active:
                token = secrets.token_urlsafe(32)
            else:
                token = None
            n = await cursor.execute('''
UPDATE workshops SET name = %s, image = %s, password = %s, active = %s, token = %s WHERE id = %s;
''',
                                     (name,
                                      post['image'],
                                      post['password'],
                                      active,
                                      token,
                                      id))
            if n == 0:
                set_message(session,
                            f'Internal error: cannot update workshop: workshop ID {id} not found.',
                            'error')
            else:
                set_message(session, f'Updated workshop {name}.', 'info')

    return web.HTTPFound(deploy_config.external_url('notebook', '/workshop-admin'))


@routes.post('/workshop-admin-delete')
@check_csrf_token
@web_authenticated_developers_only()
async def delete_workshop(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']

    post = await request.post()
    name = post['name']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            n = await cursor.execute('''
DELETE FROM workshops WHERE name = %s;
''', name)

    session = await aiohttp_session.get_session(request)
    if n == 1:
        set_message(session, f'Deleted workshop {name}.', 'info')
    else:
        set_message(session, f'Workshop {name} not found.', 'error')

    return web.HTTPFound(deploy_config.external_url('notebook', '/workshop-admin'))


workshop_routes = web.RouteTableDef()


@workshop_routes.get('')
@workshop_routes.get('/')
@web_maybe_authenticated_workshop_guest
async def workshop_get_index(request, userdata):
    page_context = {
        'notebook_service': 'workshop'
    }
    return await render_template('workshop', request, userdata, 'workshop/index.html', page_context)


@workshop_routes.get('/login')
@web_maybe_authenticated_workshop_guest
async def workshop_get_login(request, userdata):
    if userdata:
        return web.HTTPFound(location=deploy_config.external_url('workshop', '/notebook'))

    page_context = {
        'notebook_service': 'workshop'
    }
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
            await cursor.execute('SELECT * FROM workshops WHERE name = %s', name)
            workshops = await cursor.fetchall()

    def forbidden():
        set_message(
            session,
            'No such workshop.  Check the workshop name and password and try again.',
            'error')
        raise web.HTTPFound(location=deploy_config.external_url('workshop', ''))

    if len(workshops) != 1:
        forbidden()
    workshop = workshops[0]

    if workshop['password'] != password:
        forbidden()

    # use hex since K8s labels can't start or end with _ or -
    user_id = secrets.token_hex(16)
    session['workshop_session'] = {
        'workshop_name': name,
        'workshop_token': workshop['token'],
        'workshop_image': workshop['image'],
        'id': user_id
    }

    set_message(session, f'Welcome to the {name} workshop!', 'info')

    return web.HTTPFound(location=deploy_config.external_url('workshop', '/notebook'))


@workshop_routes.post('/logout')
@check_csrf_token
@web_authenticated_workshop_guest_only
async def workshop_post_logout(request, userdata):
    app = request.app
    user_id = userdata['id']
    notebook = await get_user_notebook(app, user_id)
    if notebook:
        # Notebook is inaccessible since login creates a new random
        # user id, so delete it.
        dbpool = app['dbpool']
        k8s = app['k8s_client']
        await delete_worker_pod(k8s, notebook['pod_name'])
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    'DELETE FROM notebooks WHERE user_id = %s;', user_id)

    session = await aiohttp_session.get_session(request)
    if 'workshop_session' in session:
        del session['workshop_session']

    return web.HTTPFound(location=deploy_config.external_url('workshop', '/notebook'))


@workshop_routes.get('/notebook')
@web_authenticated_workshop_guest_only
async def workshop_get_notebook(request, userdata):
    return await _get_notebook('workshop', request, userdata)


@workshop_routes.post('/notebook')
@check_csrf_token
@web_authenticated_workshop_guest_only
async def workshop_post_notebook(request, userdata):
    return await _post_notebook('workshop', request, userdata)


@workshop_routes.get('/auth/{requested_notebook_token}')
@web_authenticated_workshop_guest_only
async def workshop_get_auth(request, userdata):
    return await _get_auth(request, userdata)


@workshop_routes.post('/notebook/delete')
@check_csrf_token
@web_authenticated_workshop_guest_only
async def workshop_delete_notebook(request, userdata):
    return await _delete_notebook('workshop', request, userdata)


@workshop_routes.get('/notebook/wait')
@web_authenticated_workshop_guest_only
async def workshop_wait_websocket(request, userdata):
    return await _wait_websocket('workshop', request, userdata)


@workshop_routes.get('/error')
@web_maybe_authenticated_user
async def workshop_get_error(request, userdata):
    return await _get_error('workshop', request, userdata)


async def on_startup(app):
    if 'BATCH_USE_KUBE_CONFIG' in os.environ:
        await config.load_kube_config()
    else:
        config.load_incluster_config()
    app['k8s_client'] = client.CoreV1Api()

    app['dbpool'] = await create_database_pool()


def run():
    sass_compile('notebook')
    root = os.path.dirname(os.path.abspath(__file__))

    # notebook
    notebook_app = web.Application()

    notebook_app.on_startup.append(on_startup)

    setup_aiohttp_jinja2(notebook_app, 'notebook')
    setup_aiohttp_session(notebook_app)

    routes.static('/static', f'{root}/static')
    setup_common_static_routes(routes)
    notebook_app.add_routes(routes)

    # workshop
    workshop_app = web.Application()

    workshop_app.on_startup.append(on_startup)

    setup_aiohttp_jinja2(workshop_app, 'notebook')
    setup_aiohttp_session(workshop_app)

    workshop_routes.static('/static', f'{root}/static')
    setup_common_static_routes(workshop_routes)
    workshop_app.add_routes(workshop_routes)

    # root app
    root_app = web.Application()
    root_app.add_domain('notebook*',
                        deploy_config.prefix_application(notebook_app, 'notebook'))
    root_app.add_domain('workshop*',
                        deploy_config.prefix_application(workshop_app, 'workshop'))
    web.run_app(root_app,
                access_log_format='%a %t "%r" %s %b "%{Host}i" "%{Referer}i" "%{User-Agent}i"',
                host='0.0.0.0', port=5000)
