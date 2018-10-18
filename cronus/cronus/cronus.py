"""
A Jupyter notebook service with local-mode Hail pre-installed
"""
from flask import Flask, session, redirect, render_template
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
fh = logging.FileHandler('cronus.log')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
log = logging.getLogger('cronus')
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
app.secret_key = uuid.uuid4().hex  # FIXME read from file
INSTANCE_ID = uuid.uuid4().hex

def start_pod(jupyter_token):
    pod_id = uuid.uuid4().hex
    pod_spec = kube.client.V1PodSpec(
        containers=[
            kube.client.V1Container(
                command=[
                    'jupyter',
                    'notebook',
                    f'--NotebookApp.token={jupyter_token}'
                ],
                name='default',
                image='gcr.io/broad-ctsa/cronus',
                ports=[kube.client.V1ContainerPort(container_port=8888)],
                resources=kube.client.V1ResourceRequirements(
                    requests={'cpu': '3.7', 'memory': '4G'}))])
    pod_template = kube.client.V1Pod(
        metadata=kube.client.V1ObjectMeta(
            generate_name='cronus-job-',
            labels={
                'app': 'cronus-job',
                'hail.is/cronus-instance': INSTANCE_ID,
                'uuid': pod_id,
            }),
        spec=pod_spec)
    pod = k8s.create_namespaced_pod('default', pod_template)
    service_spec = kube.client.V1ServiceSpec(
        selector={
            'app': 'cronus-job',
            'hail.is/cronus-instance': INSTANCE_ID,
            'uuid': pod_id},
        ports=[kube.client.V1ServicePort(port=80, target_port=8888)])
    service_template = kube.client.V1Service(
        metadata=kube.client.V1ObjectMeta(
            generate_name='cronus-job-service-',
            labels={
                'app': 'cronus-job',
                'hail.is/cronus-instance': INSTANCE_ID}),
        spec=service_spec)
    svc = k8s.create_namespaced_service('default', service_template)
    while True:
        pod = k8s.read_namespaced_pod(name=pod.metadata.name,
                                      namespace='default')
        if pod.status.phase != 'Pending':
            break
        time.sleep(1)
    return svc


def external_url_for(*args, **kwargs):
    # NOTE: nginx strips https and sets X-Forwarded-Proto: https, but
    # it is not used by request.url or url_for, so rewrite the url and
    # set _scheme='https' explicitly.
    kwargs['_scheme'] = 'https'
    kwargs['_external'] = 'true'
    return flask.url_for(*args, **kwargs)


@app.route('/healthcheck')
def healthcheck():
    return '', 200


@app.route('/')
def root():
    if 'svc_name' not in session:
        log.info(f'no svc_name found in session {session.keys()}')
        return render_template('index.html', new=external_url_for('new'))
    svc_name = session['svc_name']
    jupyter_token = session['jupyter_token']
    log.info('redirecting to ' + external_url_for('root') + f'cronus/instance/{svc_name}/?token={jupyter_token}')
    return redirect(external_url_for('root') + f'cronus/instance/{svc_name}/?token={jupyter_token}')


@app.route('/new', methods=['POST'])
def new():
    log.info('new received')
    jupyter_token = uuid.uuid4().hex  # FIXME: probably should be cryptographically secure
    svc = start_pod(jupyter_token)
    svc_name = svc.metadata.name
    session['svc_name'] = svc_name
    session['jupyter_token'] = jupyter_token
    return external_url_for('root') + f'cronus/instance/{svc_name}/?token={jupyter_token}', 200


@app.route('/auth/<requested_svc_name>')
def auth(requested_svc_name):
    approved_svc_name = session.get('svc_name')
    if approved_svc_name and approved_svc_name == requested_svc_name:
        return '', 200
    return '', 403


app.run('0.0.0.0')
