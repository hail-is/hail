import logging
import os
import uuid
from functools import wraps
import asyncio
import pymysql
import aiohttp
from aiohttp import web
import aiohttp_session
import aiohttp_session.cookie_storage
import aiohttp_jinja2
from kubernetes_asyncio import client, config
import kubernetes_asyncio as kube

from hailtop.config import get_deploy_config
from gear import setup_aiohttp_session, create_database_pool, \
    web_authenticated_users_only, web_maybe_authenticated_user, web_authenticated_developers_only, \
    new_csrf_token, check_csrf_token
from web_common import sass_compile, setup_aiohttp_jinja2, setup_common_static_routes, \
    set_message, base_context

log = logging.getLogger('notebook')

NOTEBOOK_NAMESPACE = os.environ['HAIL_NOTEBOOK_NAMESPACE']

deploy_config = get_deploy_config()

routes = web.RouteTableDef()

# Must be int for Kubernetes V1 api timeout_seconds property
KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5))

POD_PORT = 8888

DEFAULT_WORKER_IMAGE = os.environ['HAIL_NOTEBOOK_WORKER_IMAGE']

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')


async def start_pod(k8s, userdata):
    notebook_base_path = deploy_config.base_path('notebook')

    jupyter_token = uuid.uuid4().hex
    pod_id = uuid.uuid4().hex

    command = [
        'jupyter',
        'notebook',
        f'--NotebookApp.token={jupyter_token}',
        f'--NotebookApp.base_url={notebook_base_path}/instance/{pod_id}/',
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
                'uuid': pod_id,
                'jupyter-token': jupyter_token,
                'user_id': str(user_id)
            }),
        spec=pod_spec)
    pod = await k8s.create_namespaced_pod(
        NOTEBOOK_NAMESPACE,
        pod_template,
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)

    return pod


def container_status_for_ui(container_statuses):
    """
        Summarize the container status based on its most recent state

        Parameters
        ----------
        container_statuses : list[V1ContainerStatus]
            https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1ContainerStatus.md
    """
    if container_statuses is None:
        return None

    assert len(container_statuses) == 1

    state = container_statuses[0].state

    if state.running:
        return {"running": {"started_at": state.running.started_at.strftime("%Y-%m-%-d %H:%M:%S")}}

    if state.waiting:
        return {"waiting": {"reason": state.waiting.reason}}

    if state.terminated:
        return {"terminated": {
            "exit_code": state.terminated.exit_code,
            "finished_at": state.terminated.finished_at.strftime("%Y-%m-%-d %H:%M:%S"),
            "started_at": state.terminated.started_at.strftime("%Y-%m-%-d %H:%M:%S"),
            "reason": state.terminated.reason
        }}

    # FIXME
    return None


def pod_condition_for_ui(conds):
    """
        Return the most recent status=="True" V1PodCondition or None
        Parameters
        ----------
        conds : list[V1PodCondition]
            https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1PodCondition.md
    """
    if conds is None:
        return None

    maxCond = max(conds, key=lambda c: (c.last_transition_time, c.status == 'True'))

    return {"status": maxCond.status, "type": maxCond.type}


def pod_to_ui_dict(pod):
    notebook = {
        'name': 'a_notebook',
        'pod_name': pod.metadata.name,
        'pod_status': pod.status.phase,
        'pod_uuid': pod.metadata.labels['uuid'],
        'pod_ip': pod.status.pod_ip,
        'creation_date': pod.metadata.creation_timestamp.strftime("%Y-%m-%-d %H:%M:%S"),
        'jupyter_token': pod.metadata.labels['jupyter-token'],
        'container_status': container_status_for_ui(pod.status.container_statuses),
        'condition': pod_condition_for_ui(pod.status.conditions)
    }

    notebook_base_path = deploy_config.base_path('notebook')
    notebook['url'] = f"{notebook_base_path}/instance/{notebook['pod_uuid']}/?token={notebook['jupyter_token']}"

    return notebook


async def get_live_notebook(k8s, userdata):
    user_id = userdata['id']
    pods = await k8s.list_namespaced_pod(
        namespace=NOTEBOOK_NAMESPACE,
        label_selector=f"user_id={user_id}",
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)

    for pod in pods.items:
        if pod.metadata.deletion_timestamp is None:
            return pod_to_ui_dict(pod)


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
@aiohttp_jinja2.template('index.html')
@web_maybe_authenticated_user
async def index(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook')
    return context


def get_config(workshop):
    if workshop:
        return {
            'session_key': 'workshop_notebook',
            'notebook_path': '/workshop/notebook'
        }
    return {
        'session_key': 'notebook',
        'notebook_path': '/notebook'
    }


async def _get_notebook(request, userdata, workshop=False):
    config = get_config(workshop)

    k8s = request.app['k8s_client']
    notebook = await get_live_notebook(k8s, userdata)
    csrf_token = new_csrf_token()

    session = await aiohttp_session.get_session(request)
    session_key = config['session_key']
    if notebook:
        session[session_key] = notebook
    else:
        if session_key in session:
            del session[session_key]

    context = base_context(deploy_config, session, userdata, 'notebook')
    context['csrf_token'] = csrf_token
    context['notebook'] = notebook
    context['notebook_path'] = config['notebook_path']
    if workshop:
        context['workshop'] = workshop
    response = aiohttp_jinja2.render_template('notebook.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', csrf_token, secure=True, httponly=True)
    return response


async def _post_notebook(request, userdata, workshop=False):
    config = get_config(workshop)
    k8s = request.app['k8s_client']
    session = await aiohttp_session.get_session(request)
    pod = await start_pod(k8s, userdata)
    session[config['session_key']] = pod_to_ui_dict(pod)
    return web.HTTPFound(
        location=deploy_config.external_url('notebook', config['notebook_path']))


async def _delete_notebook(request, workshop=False):
    config = get_config(workshop)
    k8s = request.app['k8s_client']
    session = await aiohttp_session.get_session(request)
    session_key = config['session_key']
    notebook = session.get(session_key)
    if notebook:
        await delete_worker_pod(k8s, notebook['pod_name'])
        del session[session_key]

    return web.HTTPFound(location=deploy_config.external_url('notebook', config['notebook_path']))


async def _wait_websocket(request, workshop=False):
    config = get_config(workshop)
    session_key = config['session_key']
    session = await aiohttp_session.get_session(request)
    notebook = session.get(session_key)

    if not notebook:
        return web.HTTPNotFound()

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pod_name = notebook['pod_name']
    if notebook['pod_ip']:
        ready_url = deploy_config.external_url('notebook', f'/instance/{notebook["pod_uuid"]}/?token={notebook["jupyter_token"]}')
        attempts = 0
        while attempts < 10:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    async with session.get(ready_url, cookies=request.cookies) as resp:
                        if resp.status >= 200 and resp.status < 300:
                            log.info(f'HEAD on jupyter pod {pod_name} succeeded: {resp}')
                            break
                        else:
                            log.info(f'HEAD on jupyter pod {pod_name} failed: {resp}')
            except aiohttp.ServerTimeoutError:
                log.info(f'HEAD on jupyter pod {pod_name} timed out')

            await asyncio.sleep(1)
            attempts += 1
    else:
        k8s = request.app['k8s_client']

        log.info(f'jupyter pod {pod_name} no IP')
        attempts = 0
        while attempts < 10:
            try:
                pod = await k8s.read_namespaced_pod(
                    name=pod_name,
                    namespace=NOTEBOOK_NAMESPACE,
                    _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
                if pod.status.pod_ip:
                    log.info(f'jupyter pod {pod_name} IP {pod.status.pod_ip}')
                    break
            except Exception:  # pylint: disable=broad-except
                log.exception('while getting jupyter pod {pod_name} status')
            await asyncio.sleep(1)
            attempts += 1

    await ws.send_str("1")

    return ws


@routes.get('/notebook')
@web_authenticated_users_only()
async def get_notebook(request, userdata):
    return await _get_notebook(request, userdata)


@routes.post('/notebook/delete')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def delete_notebook(request, userdata):  # pylint: disable=unused-argument
    return await _delete_notebook(request)


@routes.post('/notebook')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def post_notebook(request, userdata):
    return await _post_notebook(request, userdata)


@routes.get('/auth/{requested_pod_uuid}')
@web_authenticated_users_only()
async def auth(request, userdata):  # pylint: disable=unused-argument
    request_pod_uuid = request.match_info['requested_pod_uuid']
    session = await aiohttp_session.get_session(request)

    notebook = session.get('notebook')
    if notebook and notebook['pod_uuid'] == request_pod_uuid:
        return web.Response(headers={
            'pod_ip': f"{notebook['pod_ip']}:{POD_PORT}"
        })

    workshop_notebook = session.get('workshop_notebook')
    if workshop_notebook and workshop_notebook['pod_uuid'] == request_pod_uuid:
        return web.Response(headers={
            'pod_ip': f"{workshop_notebook['pod_ip']}:{POD_PORT}"
        })

    return web.HTTPNotFound()


@routes.get('/worker-image')
async def worker_image(request):  # pylint: disable=unused-argument
    # FIXME return active workshop images
    return web.Response(text=DEFAULT_WORKER_IMAGE)


@routes.get('/wait')
@web_authenticated_users_only(redirect=False)
async def wait_websocket(request, userdata):  # pylint: disable=unused-argument
    return _wait_websocket(request)


@routes.get('/error')
@aiohttp_jinja2.template('error.html')
@web_maybe_authenticated_user
async def error_page(request, userdata):
    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook')
    context['error'] = request.args.get('err')
    return context


@routes.get('/user')
@aiohttp_jinja2.template('user.html')
@web_authenticated_users_only()
async def user_page(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook')
    return context


@routes.get('/workshop-admin')
@web_authenticated_developers_only()
async def workshop_admin(request, userdata):
    app = request.app
    dbpool = app['dbpool']
    csrf_token = new_csrf_token()

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM workshops')
            workshops = await cursor.fetchall()

    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook')
    context['csrf_token'] = csrf_token
    context['workshops'] = workshops
    response = aiohttp_jinja2.render_template('workshop-admin.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', csrf_token, secure=True, httponly=True)
    return response


@routes.post('/workshop-admin-create')
@check_csrf_token
@web_authenticated_developers_only()
async def create_workshop(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']
    session = await aiohttp_session.get_session(request)

    post = await request.post()
    name = post['name']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                active = (post.get('active') == 'on')
                if active:
                    token = uuid.uuid4().hex
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
                token = uuid.uuid4().hex
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


async def get_workshop_userdata(request):
    session = await aiohttp_session.get_session(request)

    if 'workshop_session' not in session:
        return None
    userdata = session['workshop_session']

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
        return await fun(request, await get_workshop_userdata(request), *args, **kwargs)
    return wrapped


def web_authenticated_workshop_guest(fun):
    @web_maybe_authenticated_workshop_guest
    @wraps(fun)
    async def wrapped(request, userdata, *args, **kwargs):
        if not userdata:
            raise web.HTTPFound(deploy_config.external_url('notebook', '/workshop/login'))
        return await fun(request, userdata, *args, **kwargs)
    return wrapped


@routes.get('/workshop')
@web_maybe_authenticated_workshop_guest
async def get_workshop_index(request, userdata):  # pylint: disable=unused-argument
    if userdata:
        return web.HTTPFound(location=deploy_config.external_url('notebook', '/workshop/notebook'))
    return web.HTTPFound(location=deploy_config.external_url('notebook', '/workshop/login'))


@routes.get('/workshop/login')
@web_maybe_authenticated_workshop_guest
async def get_workshop_login(request, userdata):
    if userdata:
        return web.HTTPFound(location=deploy_config.external_url('notebook', '/workshop/notebook'))

    csrf_token = new_csrf_token()

    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook')
    context['csrf_token'] = csrf_token
    context['workshop'] = True
    response = aiohttp_jinja2.render_template('workshop/login.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', csrf_token, secure=True, httponly=True)
    return response


@routes.post('/workshop/login')
@check_csrf_token
async def post_workshop_login(request):
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
        raise web.HTTPFound(location=deploy_config.external_url('notebook', '/workshop'))

    if len(workshops) != 1:
        forbidden()
    workshop = workshops[0]

    if workshop['password'] != password:
        forbidden()

    session['workshop_session'] = {
        'workshop_name': name,
        'workshop_token': workshop['token'],
        'workshop_image': workshop['workshop_image'],
        'id': uuid.uuid4().hex
    }

    set_message(session, f'Welcome to the {name} workshop!', 'info')

    return web.HTTPFound(location=deploy_config.external_url('notebook', '/workshop/notebook'))


@routes.post('/workshop/logout')
@check_csrf_token
async def post_workshop_logout(request):
    session = await aiohttp_session.get_session(request)
    if 'workshop_session' in session:
        del session['workshop_session']

    # Notebook is inaccessible since login creates a new random user
    # id, so delete it.
    if 'workshop_notebook' in session:
        k8s = request.app['k8s_client']
        notebook = session['workshop_notebook']
        await delete_worker_pod(k8s, notebook['pod_name'])
        del session['workshop_session']

    return web.HTTPFound(location=deploy_config.external_url('notebook', '/workshop/notebook'))


@routes.get('/workshop/notebook')
@web_authenticated_workshop_guest
async def get_workshop_notebook(request, userdata):
    return await _get_notebook(request, userdata, workshop=True)


@routes.post('/workshop/notebook')
@check_csrf_token
@web_authenticated_workshop_guest
async def post_workshop_notebook(request, userdata):
    return await _post_notebook(request, userdata, workshop=True)


@routes.post('/workshop/notebook/delete')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def delete_workshop_notebook(request, userdata):  # pylint: disable=unused-argument
    return await _delete_notebook(request, workshop=True)


@routes.post('/workshop/wait')
@web_authenticated_workshop_guest
async def workshop_wait_websocket(request, userdata):  # pylint: disable=unused-argument
    return _wait_websocket(request, workshop=True)


async def on_startup(app):
    if 'BATCH_USE_KUBE_CONFIG' in os.environ:
        await config.load_kube_config()
    else:
        config.load_incluster_config()
    app['k8s_client'] = client.CoreV1Api()

    app['dbpool'] = await create_database_pool()


def run():
    app = web.Application()

    setup_aiohttp_jinja2(app, 'notebook')
    setup_aiohttp_session(app)

    sass_compile('notebook')
    root = os.path.dirname(os.path.abspath(__file__))
    routes.static('/static', f'{root}/static')
    setup_common_static_routes(routes)
    app.add_routes(routes)

    app.on_startup.append(on_startup)
    web.run_app(deploy_config.prefix_application(app, 'notebook'), host='0.0.0.0', port=5000)
