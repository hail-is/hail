import aiohttp
import aiohttp_jinja2
import aiohttp_session
import aiohttp_session.cookie_storage
import asyncio
import base64
import jinja2
import kubernetes_asyncio as kube
import logging
import os
import re
import uuid
import uvloop

from cryptography import fernet
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['funcNameAndLine'] = "{}:{}".format(record.funcName, record.lineno)


def configure_logging():
    fmt = CustomJsonFormatter('(levelname) (asctime) (filename) (funcNameAndLine) (message)')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logging.basicConfig(handlers=[stream_handler], level=logging.INFO)


configure_logging()
log = logging.getLogger('notebook')

app = aiohttp.web.Application(client_max_size=None)
routes = aiohttp.web.RouteTableDef()

uvloop.install()


def read_string(f):
    with open(f, 'r') as f:
        return f.read().strip()


app.secret_key = read_string('/notebook-secrets/secret-key')
PASSWORD = read_string('/notebook-secrets/password')
ADMIN_PASSWORD = read_string('/notebook-secrets/admin-password')
INSTANCE_ID = uuid.uuid4().hex

log.info(f'INSTANCE_ID {INSTANCE_ID}')

try:
    with open('notebook-worker-images', 'r') as f:
        def get_name(line):
            return re.search("/([^/:]+):", line).group(1)
        WORKER_IMAGES = {get_name(line): line.strip() for line in f}
except FileNotFoundError as e:
    raise ValueError(
        "working directory must contain a file called `notebook-worker-images' "
        "containing the name of the docker image to use for worker pods.") from e


async def start_pod(jupyter_token, image):
    pod_id = uuid.uuid4().hex
    service_spec = kube.client.V1ServiceSpec(
        selector={
            'app': 'notebook-worker',
            'hail.is/notebook-instance': INSTANCE_ID,
            'uuid': pod_id},
        ports=[kube.client.V1ServicePort(port=80, target_port=8888)])
    service_template = kube.client.V1Service(
        metadata=kube.client.V1ObjectMeta(
            generate_name='notebook-worker-service-',
            labels={
                'app': 'notebook-worker',
                'hail.is/notebook-instance': INSTANCE_ID,
                'uuid': pod_id}),
        spec=service_spec)
    svc = await app['k8s'].create_namespaced_service(
        'default',
        service_template
    )
    pod_spec = kube.client.V1PodSpec(
        security_context=kube.client.V1SecurityContext(
            run_as_user=1000),
        containers=[
            kube.client.V1Container(
                command=[
                    'jupyter',
                    'notebook',
                    f'--NotebookApp.token={jupyter_token}',
                    f'--NotebookApp.base_url=/instance/{svc.metadata.name}/'
                ],
                name='default',
                image=image,
                ports=[kube.client.V1ContainerPort(container_port=8888)],
                resources=kube.client.V1ResourceRequirements(
                    requests={'cpu': '1.601', 'memory': '1.601G'}),
                readiness_probe=kube.client.V1Probe(
                    http_get=kube.client.V1HTTPGetAction(
                        path=f'/instance/{svc.metadata.name}/login',
                        port=8888)))])
    pod_template = kube.client.V1Pod(
        metadata=kube.client.V1ObjectMeta(
            generate_name='notebook-worker-',
            labels={
                'app': 'notebook-worker',
                'hail.is/notebook-instance': INSTANCE_ID,
                'uuid': pod_id,
            }),
        spec=pod_spec)
    pod = await app['k8s'].create_namespaced_pod(
        'default',
        pod_template)
    return svc, pod


@routes.get('/healthcheck')
async def healthcheck():
    return aiohttp.web.Response()


@routes.get('/', name='root')
@aiohttp_jinja2.template('index.html')
async def root(request):
    session = await aiohttp_session.get_session(request)
    if 'svc_name' not in session:
        log.info(f'no svc_name found in session {session.keys()}')
        return {'form_action_url': str(request.app.router['new'].url_for()),
                'images': list(WORKER_IMAGES),
                'default': 'gew2019'}
    svc_name = session['svc_name']
    jupyter_token = session['jupyter_token']
    # str(request.app.router['root'].url_for()) +
    url = request.url.with_path(f'instance/{svc_name}/?token={jupyter_token}')
    log.info('redirecting to ' + url)
    raise aiohttp.web.HTTPFound(url)


@routes.get('/new', name='new')
async def new_get(request):
    session = await aiohttp_session.get_session(request)
    pod_name = session.get('pod_name')
    svc_name = session.get('svc_name')
    if pod_name:
        await delete_worker_pod(pod_name, svc_name)
    session.clear()
    raise aiohttp.web.HTTPFound(
        request.app.router['root'].url_for())


@routes.post('/new')
async def new_post(request):
    session = await aiohttp_session.get_session(request)
    log.info('new received')
    form = await request.post()
    password = form['password']
    image = form['image']
    if password != PASSWORD or image not in WORKER_IMAGES:
        raise aiohttp.web.HTTPForbidden()
    jupyter_token = fernet.Fernet.generate_key().decode('ascii')
    svc, pod = await start_pod(jupyter_token, WORKER_IMAGES[image])
    session['svc_name'] = svc.metadata.name
    session['pod_name'] = pod.metadata.name
    session['jupyter_token'] = jupyter_token
    raise aiohttp.web.HTTPFound(
        request.app.router['wait'].url_for())


@routes.get('/wait', name='wait')
@aiohttp_jinja2.template('wait.html')
async def wait_webpage(request):
    return {}


@routes.get('/auth/{requested_svc_name}')
async def auth(request):
    session = await aiohttp_session.get_session(request)
    requested_svc_name = request.match_info['requested_svc_name']
    approved_svc_name = session.get('svc_name')
    if approved_svc_name and approved_svc_name == requested_svc_name:
        return aiohttp.web.Response()
    raise aiohttp.web.HTTPForbidden()


async def get_all_workers():
    workers = await app['k8s'].list_namespaced_pod(
        namespace='default',
        watch=False,
        label_selector='app=notebook-worker')
    workers_and_svcs = []
    for w in workers.items:
        uuid = w.metadata.labels['uuid']
        svcs = await app['k8s'].list_namespaced_service(
            namespace='default',
            watch=False,
            label_selector='uuid=' + uuid).items
        assert len(svcs) <= 1
        if len(svcs) == 1:
            workers_and_svcs.append((w, svcs[0]))
        else:
            log.info(f'assuming pod {w.metadata.name} is getting deleted '
                     f'because it has no service')
    return workers_and_svcs


@routes.get('/workers', name='workers')
@aiohttp_jinja2.template('workers.html')
async def workers(request):
    session = await aiohttp_session.get_session(request)
    if not session.get('admin'):
        raise aiohttp.web.HTTPFound(
            request.app.router['admin-login'].url_for())
    workers_and_svcs = await get_all_workers()
    return {'workers': workers_and_svcs,
            'workers_url': str(request.app.router['workers'].url_for()),
            'leader_instance': INSTANCE_ID}


@routes.get('/workers/{pod_name}/{svc_name}/delete')
async def workers_delete(request):
    session = await aiohttp_session.get_session(request)
    pod_name = request.match_info['pod_name']
    svc_name = request.match_info['svc_name']
    if not session.get('admin'):
        raise aiohttp.web.HTTPFound(
            request.app.router['admin-login'].url_for())
    await delete_worker_pod(pod_name, svc_name)
    raise aiohttp.web.HTTPFound(request.app.router['workers'].url_for())


@routes.post('/workers/delete-all-workers')
async def delete_all_workers(request):
    session = await aiohttp_session.get_session(request)
    if not session.get('admin'):
        raise aiohttp.web.HTTPFound(
            request.app.router['admin-login'].url_for())
    workers_and_svcs = await get_all_workers()
    await asyncio.gather(*[
        delete_worker_pod(pod.metadata.name, svc.metadata.name)
        for pod, svc in workers_and_svcs])
    raise aiohttp.web.HTTPFound(request.app.router['workers'].url_for())


async def delete_worker_pod(pod_name, svc_name):
    try:
        await app['k8s'].delete_namespaced_pod(
            pod_name,
            'default')
    except kube.client.rest.ApiException as e:
        log.info(f'pod {pod_name} or associated service already deleted {e}')
    try:
        await app['k8s'].delete_namespaced_service(
            svc_name,
            'default')
    except kube.client.rest.ApiException as e:
        log.info(f'service {svc_name} (for pod {pod_name}) already deleted {e}')


@routes.get('/admin-login', name='admin-login')
@aiohttp_jinja2.template('admin-login.html')
async def admin_login(request):
    return {'form_action_url': str(request.app.router['workers'].url_for())}


@routes.post('/admin-login')
async def admin_login_post(request):
    session = await aiohttp_session.get_session(request)
    form = await request.post()
    if form['password'] != ADMIN_PASSWORD:
        raise aiohttp.web.HTTPForbidden()
    session['admin'] = True
    raise aiohttp.web.HTTPFound(request.app.router['workers'].url_for())


@routes.get('/worker-image')
async def worker_image(request):
    del request
    return aiohttp.web.Response(text='\n'.join(WORKER_IMAGES.values()))


@routes.get('/waitws')
async def wait_websocket(request):
    session = await aiohttp_session.get_session(request)
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    pod_name = session['pod_name']
    svc_name = session['svc_name']
    jupyter_token = session['jupyter_token']
    log.info(f'received wait websocket for {svc_name} {pod_name}')
    while True:
        try:
            response = await app['client_session'].head(
                f'https://notebook.hail.is/instance-ready/{svc_name}/',
                timeout=1)
            if response.status < 500:
                log.info(
                    f'HEAD on jupyter succeeded for {svc_name} {pod_name} '
                    f'response: {response}')
                # if someone responds with a 2xx, 3xx, or 4xx, the notebook
                # server is alive and functioning properly (in particular, our
                # HEAD request will return 405 METHOD NOT ALLOWED)
                break
            # somewhat unusual, means the gateway had an error before we
            # timed out, usually means the gateway itself is broken
            log.info(f'HEAD on jupyter failed for {svc_name} {pod_name} response: {response}')
        except Exception as e:
            log.info(f'GET on jupyter failed for {svc_name} {pod_name} {e}')
        await asyncio.sleep(1)
    notebook_url_scheme = request.url.scheme.replace('ws', 'http')
    notebook_url = request.url.with_scheme(notebook_url_scheme)
    notebook_url = notebook_url.with_path(f'instance/{svc_name}/')
    notebook_url = notebook_url.with_query(token=jupyter_token)
    await ws.send_str(str(notebook_url))
    await ws.close()
    log.info(f'notification sent to user for {svc_name} {pod_name}')
    return ws


async def setup_k8s(app):
    kube.config.load_incluster_config()
    app['k8s'] = kube.client.CoreV1Api()


async def cleanup(app):
    await app['client_session'].close()


if __name__ == '__main__':
    my_path = os.path.dirname(os.path.abspath(__file__))
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader(os.path.join(my_path, 'templates')))
    routes.static('/static', os.path.join(my_path, 'static'))
    app.add_routes(routes)
    app.on_startup.append(setup_k8s)
    app['client_session'] = aiohttp.ClientSession()
    app.on_cleanup.append(cleanup)
    fernet_key = fernet.Fernet.generate_key()
    secret_key = base64.urlsafe_b64decode(fernet_key)
    aiohttp_session.setup(
        app,
        aiohttp_session.cookie_storage.EncryptedCookieStorage(secret_key))
    aiohttp.web.run_app(app, host='0.0.0.0', port=5000)
