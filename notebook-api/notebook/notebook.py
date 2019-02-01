"""
A Jupyter notebook service with local-mode Hail pre-installed
"""
import gevent
# must happen before anytyhing else
from gevent import monkey
monkey.patch_all()

import base64
import requests
import ujson
from flask import Flask, session, redirect, render_template, request
from flask_sockets import Sockets
import flask
import kubernetes as kube
import logging
import os
import re
import requests
import time
import uuid
import pymysql
from dotenv import load_dotenv

load_dotenv(verbose=True)

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


# TODO: hash instead?
def UNSAFE_user_id_transform(user_id): return user_id.replace('|', '--_--')


DOMAIN = os.environ.get("DOMAIN", "http://notebook-api")
AUTH_GATEWAY = os.environ.get("AUTH_GATEWAY", "http://auth-gateway")
HAIL_JUPYTER_IMAGE = os.environ.get(
    "HAIL_JUPYTER_IMAGE", "gcr.io/hail-vdc/hail-jupyter")

KUBERNETES_TIMEOUT_IN_SECONDS = float(
    os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
app.secret_key = read_string('/notebook-secrets/secret-key')
PASSWORD = read_string('/notebook-secrets/password')
ADMIN_PASSWORD = read_string('/notebook-secrets/admin-password')
INSTANCE_ID = uuid.uuid4().hex

log.info(f'DOMAIN: {DOMAIN}')
# used for /verify; will likely go away once 2nd (notebook) verify step moved to nginx
log.info(f'AUTH_GATEWAY: {AUTH_GATEWAY}')
log.info(f'HAIL_JUPYTER_IMAGE: {HAIL_JUPYTER_IMAGE}')
log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS: {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'INSTANCE_ID: {INSTANCE_ID}')

try:
    with open('notebook-worker-images', 'r') as f:
        def get_name(line):
            return re.search("/([^/:]+):", line).group(1)
        WORKER_IMAGES = {get_name(line): line.strip() for line in f}
except FileNotFoundError as e:
    raise ValueError(
        "working directory must contain a file called `notebook-worker-images' "
        "containing the name of the docker image to use for worker pods.") from e


def start_pod(jupyter_token, image, labels={}):
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
                'uuid': pod_id,
                **labels}),
        spec=service_spec)
    svc = k8s.create_namespaced_service(
        'default',
        service_template,
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS
    )
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
                'svc_name': svc.metadata.name,
                **labels
            },),
        spec=pod_spec)
    pod = k8s.create_namespaced_pod(
        'default',
        pod_template,
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS,
    )

    # Avoid extra queries to get pod_name (or reciprocol, svc_name)
    # svc.metadata.labels['pod_name'] = pod.metadata.name
    # k8s.patch_namespaced_service(svc.metadata.name, 'default', {
    #     'metadata': {
    #         'labels': {
    #             'pod_name': pod.metadata.name
    #         }
    #     }
    # })
    return svc, pod


def forbidden():
    return 'Forbidden', 403

# Have consumer pass a graph to awalk

# TODO: could support array values as well, i.e path.to.0


def get_path(data, path: str):
    idx = path.find('.')
    if idx == -1:
        if isinstance(data, dict):
            return data[path]
        return getattr(data, path)

    return get_path(getattr(data, path[0:idx]), path[idx + 1:])

# Could make this somewhat more elegenat by checking whether resources is list or object
# if object, return json object, else json array; this may come at 1 additional functional call per resource
# or an uglier inline block
# Given a kubernetes object, and some collection of field paths, walk path tree and fetch values
# @param resources<List> : kubernetes v1 object
# @param paths<List<List[3]>> : fields and transformations: [key, path_in_kube_object, lambda_for_found_val]


def marshall_json(resources: [], paths=[], flatten=False):
    if len(resources) == 0 or len(paths) == 0:
        if flatten == True:
            return "{}"

        return "[]"

    resp = []
    for rsc in resources:
        data = {}
        for path in paths:
            if len(path) == 3:
                data[path[0]] = path[2](get_path(rsc, path[1]))
            else:
                data[path[0]] = get_path(rsc, path[1])

        resp.append(data)

    if flatten == True and len(resources) == 1:
        return ujson.dumps(resp[0])

    return ujson.dumps(resp)


# Services and pods are seperate resources, and operations between them are not atomic
# Must support deletion of one indep of presence other


def del_pod(pod_name):
    k8s.delete_namespaced_pod(
        pod_name,
        'default',
        kube.client.V1DeleteOptions(),
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)


def del_svc(svc_name):
    k8s.delete_namespaced_service(
        svc_name,
        'default',
        kube.client.V1DeleteOptions(),
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)


##### New notebook methods meant for consumption from agnostic client#####
###### Methods that do not call the auth-gateway /verify method must not be directly public #######

# TODO: learn how to properly handle webscocket close in gevent + wsgi
# or just move to aiohttp and stop dealing with greenlets and websockets with
# no bound events (though there may be a way in gevent's websocket package + wsgi)
# simply checkingfo ws.close isn't enough
# https://github.com/heroku-python/flask-sockets/issues/60
# A reactive approach to websockets. Primary benefit is 1 connection per user
# rather than 1 connection per pod
# no need to go to public web to hit http endpoint
# and real insight into pod status
# Weakness is currently not checking whether svc is accessible; easily can add


@sockets.route('/api/ws')
def echo_socket(ws):
    user_id = ws.environ['HTTP_USER']  # and scope is ws.environ['HTTP_SCOPE']

    k8s = kube.client.CoreV1Api()

    w = kube.watch.Watch()

    for event in w.stream(k8s.list_namespaced_pod, namespace='default', watch=False,
                          label_selector=f"user_id={UNSAFE_user_id_transform(user_id)}"):

        # This won't prevent socket is dead errors, presumably something related to
        # greenlet socket handling and our inability to add an on_closed callback to end watch
        if ws.closed:
            return

        try:
            obj = event["object"]
            print("Event: %s %s %s" %
                  (event['type'], event['object'].kind, event['object'].metadata.name))

            if event['type'] == 'MODIFIED' or event['type'] == 'ADDED':
                ws.send(
                    f'{{"event": "{event["type"]}", "resource": {marshall_json([obj], pod_paths, True)}}}')

            if event['type'] == 'DELETED':

                ws.send(
                    f'{{"event": "DELETED", "resource": {marshall_json([obj], pod_paths, True)}}}')
        except Exception as e:
            log.error(e)
            return

    while not ws.closed:
        gevent.sleep(10)


@app.route('/api/verify/<svc_name>/', methods=['GET'])
def verify(svc_name):
    access_token = request.args.get('authorization')
    token = request.args.get('token')

    if not access_token:
        return forbidden()

    resp = requests.get(f'{AUTH_GATEWAY}/verify',
                        headers={'Authorization': f'Bearer {access_token}'})

    if resp.status_code != 200:
        return forbidden()

    user_id = resp.headers.get('User')

    if not user_id:
        return forbidden()

    res = k8s.read_namespaced_service(svc_name, 'default')

    l = res.metadata.labels
    if l['jupyter_token'] != token or l['user_id'] != UNSAFE_user_id_transform(user_id):
        return forbidden()

    return '', 200


# read_svc_status checks that the pod's backing service exists
# It returns a string status, which must match the corresponding Pod status
# Ex: If Kubernetes names running pod status (phase) "Running", and svc is running
# we should return "Running" here
# TODO: Inspect exceptions to distinguish between 404 and others

def read_svc_status(svc_name):
    try:
        svc = k8s.read_namespaced_service(svc_name, 'default')
        return 'Running'
    except:
        return 'Deleted'


# TODO: support lambda expressions
# TODO: had default key as path name, but this adds branches/if statements
# in tight loop, dont think the space savings worth it

pod_paths = [
    ['name', 'metadata.labels.name'],
    ['pod_name', 'metadata.name'],
    ['svc_name', 'metadata.labels.svc_name'],
    ['pod_status', 'status.phase'],
    ['svc_status', 'metadata.labels.svc_name', lambda x: read_svc_status(x)],
    ['creation_date', 'metadata.creation_timestamp',
        lambda x: x.strftime('%D')],
    ['token', 'metadata.labels.jupyter_token']
]


@app.route('/api', methods=['GET'])
def get_notebooks():
    user_id = request.headers.get('User')

    if not user_id:
        return forbidden()

    # svcs = k8s.list_namespaced_service(
    #     namespace='default',
    #     watch=False,
    #     label_selector='user_id=' + UNSAFE_user_id_transform(user_id),
    #     _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS).items
    # k8s.read_namespaced_service(svc_name, 'default')
    pods = k8s.list_namespaced_pod(
        namespace='default', watch=False,
        label_selector=f"user_id={UNSAFE_user_id_transform(user_id)}").items
    print(pods)
    return marshall_json(pods, pod_paths), 200

# TODO: decide if need to communicate issues to user; probably just alert devs


@app.route('/api/<svc_name>', methods=['DELETE'])
def delete_notebook(svc_name):
    # TODO: Is it possible to have a falsy token value and get here?
    if not request.headers.get("User") or not svc_name:
        return forbidden()

    escp_user_id = UNSAFE_user_id_transform(request.headers.get("User"))

    try:
        svc = k8s.read_namespaced_service(svc_name, 'default')

        if svc.metadata.labels['user_id'] != escp_user_id:
            return forbidden()

        del_svc(svc_name)

        pods = k8s.list_namespaced_pod(
            namespace='default', watch=False, label_selector=f"uuid={svc.metadata.labels['uuid']}").items

        if len(pods) == 0:
            log.error(
                f"svc_name: {svc_name} pod_uuid: {svc.metadata.labels['uuid']}: pod not found for given uuid")
            return '', 200

        if len(pods) > 1:
            log.error(
                f"svc_name: {svc_name} pod_uuid: {svc.metadata.labels['uuid']}: pod_uuid is not unique")

        for pod in pods:
            if pod.metadata.labels['user_id'] != escp_user_id:
                return forbidden()
            del_pod(pod.metadata.name)

        return '', 200

    except kube.client.rest.ApiException as e:
        # TODO: enable this; front-end lightweight library makes it difficult to recover from trivial errors
        # There must be a nicer way to read http error code from kubernetes
        # return '', str(e)[1:4]
        return '', 200


@app.route('/api', methods=['POST'])
def new_notebook():
    name = pymysql.escape_string(request.form.get('name', 'a_notebook'))

    image = pymysql.escape_string(
        request.form.get('image', HAIL_JUPYTER_IMAGE))

    user_id = request.headers.get('User')

    if not user_id or image not in WORKER_IMAGES:
        return forbidden()

    # TODO: Do we want jupyter_token to be globally unique?
    # Token doesn't need to be crypto secure, just unique (since we authorize user)
    # However, encryption or even detection of modification via hash may further reduce attack space
    jupyter_token = uuid.uuid4().hex
    _, pod = start_pod(
        jupyter_token, WORKER_IMAGES[image],
        labels={'name': name,
                'jupyter_token': jupyter_token,
                'user_id': UNSAFE_user_id_transform(user_id)}
    )

    return marshall_json([pod], pod_paths, True), 200


if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(
        ('', 5000), app, handler_class=WebSocketHandler, log=log)
    server.serve_forever()
