import logging
import os
import uuid
import asyncio
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

log = logging.getLogger('notebook2')

NOTEBOOK_NAMESPACE = os.environ['HAIL_NOTEBOOK_NAMESPACE']

deploy_config = get_deploy_config()

routes = web.RouteTableDef()

# Must be int for Kubernetes V1 api timeout_seconds property
KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5))

INSTANCE_ID = uuid.uuid4().hex

POD_PORT = 8888

WORKER_IMAGE = os.environ['HAIL_NOTEBOOK2_WORKER_IMAGE']

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'INSTANCE_ID {INSTANCE_ID}')


async def start_pod(k8s, userdata):
    notebook2_base_path = deploy_config.base_path('notebook2')

    jupyter_token = uuid.uuid4().hex
    pod_id = uuid.uuid4().hex

    user_id = userdata['id']
    ksa_name = userdata['ksa_name']
    bucket = userdata['bucket_name']
    gsa_key_secret_name = userdata['gsa_key_secret_name']
    jwt_secret_name = userdata['jwt_secret_name']

    pod_spec = kube.client.V1PodSpec(
        service_account_name=ksa_name,
        containers=[
            kube.client.V1Container(
                command=[
                    'jupyter',
                    'notebook',
                    f'--NotebookApp.token={jupyter_token}',
                    f'--NotebookApp.base_url={notebook2_base_path}/instance/{pod_id}/',
                    f'--GoogleStorageContentManager.default_path="{bucket}"',
                    "--ip", "0.0.0.0", "--no-browser", "--allow-root"
                ],
                name='default',
                image=WORKER_IMAGE,
                env=[kube.client.V1EnvVar(name='HAIL_DEPLOY_CONFIG_FILE',
                                          value='/deploy-config/deploy-config.json')],
                ports=[kube.client.V1ContainerPort(container_port=POD_PORT)],
                resources=kube.client.V1ResourceRequirements(
                    requests={'cpu': '1.601', 'memory': '1.601G'}),
                readiness_probe=kube.client.V1Probe(
                    period_seconds=5,
                    http_get=kube.client.V1HTTPGetAction(
                        path=f'{notebook2_base_path}/instance/{pod_id}/login',
                        port=POD_PORT)),
                volume_mounts=[
                    kube.client.V1VolumeMount(
                        mount_path='/gsa-key',
                        name='gsa-key',
                        read_only=True),
                    kube.client.V1VolumeMount(
                        mount_path='/user-tokens',
                        name='user-tokens',
                        read_only=True),
                    kube.client.V1VolumeMount(
                        mount_path='/deploy-config',
                        name='deploy-config',
                        read_only=True)
                ])
        ],
        volumes=[
            kube.client.V1Volume(
                name='gsa-key',
                secret=kube.client.V1SecretVolumeSource(
                    secret_name=gsa_key_secret_name)),
            kube.client.V1Volume(
                name='user-tokens',
                secret=kube.client.V1SecretVolumeSource(
                    secret_name=jwt_secret_name)),
            kube.client.V1Volume(
                name='deploy-config',
                secret=kube.client.V1SecretVolumeSource(
                    secret_name='deploy-config'))
        ])
    pod_template = kube.client.V1Pod(
        metadata=kube.client.V1ObjectMeta(
            generate_name='notebook2-worker-',
            labels={
                'app': 'notebook2-worker',
                'hail.is/notebook2-instance': INSTANCE_ID,
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

    notebook2_base_path = deploy_config.base_path('notebook2')
    notebook['url'] = f"{notebook2_base_path}/instance/{notebook['pod_uuid']}/?token={notebook['jupyter_token']}"

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
    context = base_context(deploy_config, session, userdata, 'notebook2')
    return context


@routes.get('/notebook')
@aiohttp_jinja2.template('notebook.html')
@web_authenticated_users_only()
async def notebook_page(request, userdata):
    k8s = request.app['k8s_client']

    notebook = await get_live_notebook(k8s, userdata)
    token = new_csrf_token()

    session = await aiohttp_session.get_session(request)
    if notebook:
        session['notebook'] = notebook
    else:
        if 'notebook' in session:
            del session['notebook']

    context = base_context(deploy_config, session, userdata, 'notebook2')
    context['token'] = token
    context['notebook'] = notebook
    response = aiohttp_jinja2.render_template('notebook.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', token, secure=True, httponly=True)
    return response


@routes.post('/notebook/delete')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def notebook_delete(request, userdata):  # pylint: disable=unused-argument
    k8s = request.app['k8s_client']
    session = await aiohttp_session.get_session(request)
    notebook = session.get('notebook')
    if notebook:
        await delete_worker_pod(k8s, notebook['pod_name'])
        del session['notebook']

    return web.HTTPFound(location=deploy_config.external_url('notebook2', '/notebook'))


@routes.post('/notebook')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def notebook_post(request, userdata):
    k8s = request.app['k8s_client']
    session = await aiohttp_session.get_session(request)
    pod = await start_pod(k8s, userdata)
    session['notebook'] = pod_to_ui_dict(pod)
    return web.HTTPFound(location=deploy_config.external_url('notebook2', '/notebook'))


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

    return web.HTTPNotFound()


@routes.get('/worker-image')
async def worker_image(request):  # pylint: disable=unused-argument
    return web.Response(text=WORKER_IMAGE)


@routes.get('/wait')
@web_authenticated_users_only(redirect=False)
async def wait_websocket(request, userdata):  # pylint: disable=unused-argument
    k8s = request.app['k8s_client']
    session = await aiohttp_session.get_session(request)
    notebook = session.get('notebook')

    if not notebook:
        return web.HTTPNotFound()

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pod_name = notebook['pod_name']
    if notebook['pod_ip']:
        ready_url = deploy_config.external_url('notebook2', f'/instance-ready/{notebook["pod_uuid"]}')
        attempts = 0
        while attempts < 10:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    async with session.head(ready_url, cookies=request.cookies) as resp:
                        if resp.status == 405:
                            log.info(f'HEAD on jupyter pod {pod_name} succeeded: {resp}')
                            break
                        else:
                            log.info(f'HEAD on jupyter pod {pod_name} failed: {resp}')
            except aiohttp.ServerTimeoutError:
                log.info(f'HEAD on jupyter pod {pod_name} timed out')

            await asyncio.sleep(1)
            attempts += 1
    else:
        log.info(f'jupyter pod {pod_name} no IP')
        attempts = 0
        while attempts < 10:
            pod = await k8s.read_namespaced_pod(
                name=pod_name,
                namespace=NOTEBOOK_NAMESPACE,
                _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
            if pod.status.pod_ip:
                log.info(f'jupyter pod {pod_name} IP {pod.status.pod_ip}')
                break
            await asyncio.sleep(1)
            attempts += 1

    await ws.send_str("1")

    return ws


@routes.get('/error')
@aiohttp_jinja2.template('error.html')
@web_maybe_authenticated_user
async def error_page(request, userdata):
    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook2')
    context['error'] = request.args.get('err')
    return context


@routes.get('/user')
@aiohttp_jinja2.template('user.html')
@web_authenticated_users_only()
async def user_page(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook2')
    return context


@routes.get('/workshop')
@aiohttp_jinja2.template('workshop/index.html')
@web_maybe_authenticated_user
async def workshop_login(request, userdata):
    session = await aiohttp_session.get_session(request)
    return base_context(deploy_config, session, userdata, 'notebook2')


@routes.post('/workshop')
async def workshop_login_post(request):
    session = await aiohttp_session.get_session(request)
    dbpool = request.app['dbpool']

    post = await request.post()
    name = post['name']
    password = post['password']

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM workshops where name = %s', name)
            workshops = await cursor.fetchall()

    def forbidden():
        set_message(
            session,
            'No such workshop.  Check the workshop name and password and try again.',
            'error')
        raise web.HTTPFound(location=deploy_config.external_url('notebook2', '/workshop'))

    if len(workshops) != 1:
        forbidden()
    workshop = workshops[0]

    if workshop['password'] != password:
        forbidden()

    session['message'] = {
        'text': 'Joined workshop.',
        'type': 'info'
    }

    raise web.HTTPFound(location=deploy_config.external_url('notebook2', '/workshop'))


@routes.get('/workshop/admin')
@web_authenticated_developers_only()
async def workshop_admin(request, userdata):
    app = request.app
    dbpool = app['dbpool']
    token = new_csrf_token()

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM workshops')
            workshops = await cursor.fetchall()

    session = await aiohttp_session.get_session(request)
    context = base_context(deploy_config, session, userdata, 'notebook2')
    context['token'] = token
    context['workshops'] = workshops
    response = aiohttp_jinja2.render_template('workshop/admin.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', token, secure=True, httponly=True)
    return response


@routes.post('/workshop/create')
@check_csrf_token
@web_authenticated_developers_only()
async def create_workshop(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']

    post = await request.post()
    name = post['name']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
INSERT INTO workshops (name, image, password, active) VALUES (%s, %s, %s, %s);
''',
                                 (name,
                                  post['image'],
                                  post['password'],
                                  post.get('active') == 'on'))

    session = await aiohttp_session.get_session(request)
    set_message(session, f'Created workshop {name}.', 'info')

    return web.HTTPFound(deploy_config.external_url('notebook2', '/workshop/admin'))


@routes.post('/workshop/update')
@check_csrf_token
@web_authenticated_developers_only()
async def update_workshop(request, userdata):  # pylint: disable=unused-argument
    app = request.app
    dbpool = app['dbpool']

    post = await request.post()
    name = post['name']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            n = await cursor.execute('''
UPDATE workshops SET name = %s, image = %s, password = %s, active = %s WHERE id = %s;
''',
                                     (name,
                                      post['image'],
                                      post['password'],
                                      post.get('active') == 'on',
                                      post['id']))
            if n == 0:
                raise web.HTTPNotFound()

    session = await aiohttp_session.get_session(request)
    set_message(session, f'Updated workshop {name}.', 'info')

    return web.HTTPFound(deploy_config.external_url('notebook2', '/workshop/admin'))


@routes.post('/workshop/delete')
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
        set_message(session, f'Worksohp {name} not found.', 'error')

    return web.HTTPFound(deploy_config.external_url('notebook2', '/workshop/admin'))


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
    web.run_app(deploy_config.prefix_application(app, 'notebook2'), host='0.0.0.0', port=5000)
