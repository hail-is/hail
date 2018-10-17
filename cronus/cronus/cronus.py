"""
A Jupyter notebook service with local-mode Hail pre-installed
"""
from flask import Flask, session, redirect
import flask
import kubernetes as kube
import logging
import os
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
                ports=[kube.client.V1ContainerPort(
                    container_port=5000,
                    host_port=80)],
                resources=kube.client.V1ResourceRequirements(
                    requests={'cpu': '3.7', 'memory': '4G'}))])
    pod_template = kube.client.V1Pod(
        metadata=kube.client.V1ObjectMeta(
            generate_name='cronus-job-',
            labels={
                'app': 'cronus-job',
                'hail.is/cronus-instance': INSTANCE_ID
            }),
        spec=pod_spec)
    return k8s.create_namespaced_pod('default', pod_template)


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
    if 'pod_name' not in session:
        jupyter_token = uuid.uuid4().hex  # FIXME: probably should be cryptographically secure
        pod = start_pod(jupyter_token)
        session['pod_name'] = pod.metadata.name
        session['jupyter_token'] = jupyter_token
        print(f'pod {pod.metadata.name}, jupyter_token {jupyter_token}')
        return redirect(external_url_for('root') + f'?token={jupyter_token}')
    pod_name = session['pod_name']
    jupyter_token = session['jupyter_token']
    return redirect(external_url_for('root') + f'/instance/{pod_name}/?token={jupyter_token}')


@app.route('/auth/<requested_pod_name>')
def auth(requested_pod_name):
    approved_pod_name = session.get('pod_name')
    if approved_pod_name and approved_pod_name == requested_pod_name:
        return '', 200
    return '', 403


app.run('0.0.0.0')
