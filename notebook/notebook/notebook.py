"""
A Jupyter notebook service with local-mode Hail pre-installed
"""
from flask import Flask, session, redirect, render_template, request
from flask_sockets import Sockets
import gevent
import flask
import kubernetes as kube
import logging
import os
import time
import uuid

fmt = logging.Formatter(
   # NB: no space after levelname because WARNING is so long
   '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
   '%(message)s')

fh = logging.FileHandler('notebook.log')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)

log = logging.getLogger('notebook')
log.setLevel(logging.INFO)
logging.basicConfig(
    handlers=[fh, ch],
    level=logging.INFO)

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
k8s = kube.client.CoreV1Api()

app = Flask(__name__)
sockets = Sockets(app)

def read_string(f):
    with open(f, 'r') as f:
        return f.read().strip()

app.secret_key = read_string('/notebook-secrets/secret-key')
PASSWORD = read_string('/notebook-secrets/password')
ADMIN_PASSWORD = read_string('/notebook-secrets/admin-password')
INSTANCE_ID = uuid.uuid4().hex

log.info(f'INSTANCE_ID {INSTANCE_ID}')

try:
    with open('notebook-worker-image', 'r') as f:
        WORKER_IMAGE = f.read().strip()
except FileNotFoundError as e:
    raise ValueError(
        "working directory must contain a file called `notebook-worker-image' "
        "containing the name of the docker image to use for worker pods.") from e


def start_pod(jupyter_token):
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
    svc = k8s.create_namespaced_service('default', service_template)
    pod_spec = kube.client.V1PodSpec(
        containers=[
            kube.client.V1Container(
                command=[
                    'jupyter',
                    'notebook',
                    f'--NotebookApp.token={jupyter_token}',
                    f'--NotebookApp.base_url=/instance/{svc.metadata.name}/'
                ],
                name='default',
                image=WORKER_IMAGE,
                ports=[kube.client.V1ContainerPort(container_port=8888)],
                resources=kube.client.V1ResourceRequirements(
                    requests={'cpu': '3.001', 'memory': '4G'}),
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
    pod = k8s.create_namespaced_pod('default', pod_template)
    return svc, pod


def external_url_for(path):
    # NOTE: nginx strips https and sets X-Forwarded-Proto: https, but
    # it is not used by request.url or url_for, so rewrite the url and
    # set _scheme='https' explicitly.
    protocol = request.headers.get('X-Forwarded-Proto', None)
    url = flask.url_for('root', _scheme=protocol, _external='true')
    return url + path


@app.route('/healthcheck')
def healthcheck():
    return '', 200


@app.route('/')
def root():
    if 'svc_name' not in session:
        log.info(f'no svc_name found in session {session.keys()}')
        return render_template('index.html', form_action_url=external_url_for('new'))
    svc_name = session['svc_name']
    jupyter_token = session['jupyter_token']
    log.info('redirecting to ' + external_url_for(f'instance/{svc_name}/?token={jupyter_token}'))
    return redirect(external_url_for(f'instance/{svc_name}/?token={jupyter_token}'))


@app.route('/new', methods=['GET'])
def new_get():
    session.clear()
    return redirect(external_url_for('/'))

@app.route('/new', methods=['POST'])
def new_post():
    log.info('new received')
    password = request.form['password']
    if password != PASSWORD:
        return '403 Forbidden', 403
    jupyter_token = uuid.uuid4().hex  # FIXME: probably should be cryptographically secure
    svc, pod = start_pod(jupyter_token)
    session['svc_name'] = svc.metadata.name
    session['pod_name'] = pod.metadata.name
    session['jupyter_token'] = jupyter_token
    return render_template('wait.html')


@app.route('/auth/<requested_svc_name>')
def auth(requested_svc_name):
    approved_svc_name = session.get('svc_name')
    if approved_svc_name and approved_svc_name == requested_svc_name:
        return '', 200
    return '', 403


@app.route('/style.css')
def style():
    return render_template('style.css'), 201, {'Content-Type': 'text/css'}


@app.route('/centering.css')
def centering():
    return render_template('centering.css'), 201, {'Content-Type': 'text/css'}


@app.route('/workers')
def workers():
    if not session.get('admin'):
        return redirect(external_url_for('admin-login'))
    workers = k8s.list_pod_for_all_namespaces(watch=False,
                                              label_selector='app=notebook-worker')
    workers_and_svcs = []
    for w in workers.items:
        uuid = w.metadata.labels['uuid']
        svcs = k8s.list_service_for_all_namespaces(watch=False,
                                                   label_selector='uuid='+uuid).items
        assert len(svcs) <= 1
        if len(svcs) == 1:
            workers_and_svcs.append((w, svcs[0]))
        else:
            log.info(f'assuming pod {w.metadata.name} is getting deleted '
                     f'because it has no service')
    return render_template('workers.html',
                           workers=workers_and_svcs,
                           workers_url=external_url_for('workers'),
                           leader_instance=INSTANCE_ID)


@app.route('/workers/<pod_name>/delete')
def workers_delete(pod_name):
    if not session.get('admin'):
        return redirect(external_url_for('admin-login'))
    pod = k8s.read_namespaced_pod(pod_name, 'default')
    uuid = pod.metadata.labels['uuid']
    k8s.delete_namespaced_pod(pod_name, 'default', kube.client.V1DeleteOptions())
    svcs = k8s.list_service_for_all_namespaces(watch=False,
                                               label_selector='uuid='+uuid).items
    assert(len(svcs) == 1)
    k8s.delete_namespaced_service(svcs[0].metadata.name, 'default', kube.client.V1DeleteOptions())
    return redirect(external_url_for('workers'))


@app.route('/admin-login', methods=['GET'])
def admin_login():
    return render_template('admin-login.html',
                           form_action_url=external_url_for('admin-login'))


@app.route('/admin-login', methods=['POST'])
def admin_login_post():
    if request.form['password'] != ADMIN_PASSWORD:
        return '403 Forbidden', 403
    session['admin'] = True
    return redirect(external_url_for('workers'))


@sockets.route('/wait')
def wait(ws):
    pod_name = session['pod_name']
    svc_name = session['svc_name']
    jupyter_token = session['jupyter_token']
    log.info(f'received wait websocket for {svc_name} {pod_name}')
    while True:
        endpoints = k8s.read_namespaced_endpoints(name=svc_name, namespace='default')
        if endpoints.subsets and all(subset.addresses for subset in endpoints.subsets):
            log.info(f'{svc_name} {pod_name} is ready! {endpoints.subsets}')
            break
        log.info(f'{svc_name} {pod_name} not ready! {endpoints.subsets}')
        # FIXME, ERRORS?
        gevent.sleep(1)
    log.info(f'wait finished for {svc_name} {pod_name}')
    ws.send(external_url_for(f'instance/{svc_name}/?token={jupyter_token}'))


if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler, log=log)
    server.serve_forever()

