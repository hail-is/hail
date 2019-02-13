"""
A Jupyter notebook service with Hail pre-installed
It is expected that all endpoints are protected
by an upstream service that verifies an OAuth2 access_token
"""

import gevent
# must happen before anytyhing else
from gevent import monkey, pywsgi
from geventwebsocket.handler import WebSocketHandler
monkey.patch_all()

import requests
import ujson
from flask import Flask, request, Response
from flask_sockets import Sockets
import flask
import kubernetes as kube
import logging
import os
import re
import requests
import time
import uuid

try:
    with open('notebook-worker-images', 'r') as f:
        def get_name(line):
            return re.search("/([^/:]+):", line).group(1)
        WORKER_IMAGES = {get_name(line): line.strip() for line in f}
except FileNotFoundError as e:
    raise ValueError(
        "working directory must contain a file called `notebook-worker-images' "
        "containing the name of the docker image to use for worker pods.") from e

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()

DEFAULT_HAIL_IMAGE = os.environ.get("DEFAULT_HAIL_IMAGE",
                                    list(WORKER_IMAGES.keys())[0])
AUTH_GATEWAY = os.environ.get("AUTH_GATEWAY", "http://auth-gateway")
KUBERNETES_TIMEOUT_IN_SECONDS = int(
    os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5))

INSTANCE_ID = uuid.uuid4().hex

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

log.info(f'AUTH_GATEWAY: {AUTH_GATEWAY}')
log.info(f'DEFAULT_HAIL_IMAGE: {DEFAULT_HAIL_IMAGE}')
log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS: {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'INSTANCE_ID: {INSTANCE_ID}')

# Prevent issues with many websockets leading to many kube connections
# TODO: tune this, potentially watch many users with one connection
kube.config.connection_pool_maxsize = 5000
k8s = kube.client.CoreV1Api()

app = Flask(__name__)
sockets = Sockets(app)


# FIXME: use hash
def UNSAFE_user_id_transform(user_id): return user_id.replace('|', '--_--')


# TODO: decide if allowing empty strings (or whitespace-only)
# opens us to any attacks
def is_falsy_str(s: str):
    if s is None or s.strip() == "":
        return True
    return False


def forbidden():
    return 'Forbidden', 404


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
                    "--ip", "0.0.0.0", "--no-browser",
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

    return svc, pod


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


# General resource marshalling functions #


def get_path(data, path: str):
    # Not using tail recursion because python doesn't optimize such calls
    while True:
        idx = path.find('.')

        if idx == -1:
            if isinstance(data, dict):
                return data[path]
            return getattr(data, path)

        data = getattr(data, path[0:idx])

        path = path[idx + 1:]


def marshall_json(resources: [], paths=[], flatten=False):
    if len(resources) == 0 or len(paths) == 0:
        if flatten:
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

    if flatten and len(resources) == 1:
        return ujson.dumps(resp[0])

    return ujson.dumps(resp)


def read_svc_status(svc_name):
    try:
        # TODO: inspect exception for non-404
        _ = k8s.read_namespaced_service(svc_name, 'default')
        return 'Running'
    except Exception:
        return 'Deleted'


def read_containers_status(container_statuses):
    """
        Summarize the container status based on its most recent state

        Parameters
        ----------
        container_statuses : list[V1ContainerStatus]
            https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1ContainerStatus.md
    """
    if container_statuses is None:
        return None

    state = container_statuses[0].state
    rn = None
    wt = None
    tm = None
    if state.running:
        rn = {"started_at": state.running.started_at}

    if state.waiting:
        wt = {"reason": state.waiting.reason}

    if state.terminated:
        tm = {"exit_code": state.terminated.exit_code, "finished_at": state.terminated.finished_at,
              "started_at": state.terminated.started_at, "reason": state.terminated.reason}

    if rn is None and wt is None and tm is None:
        return None

    return {"running": rn, "terminated": tm, "waiting": wt}


def read_conditions(conds):
    """
        Return the most recent status=="True" V1PodCondition or None

        Parameters
        ----------
        conds : list[V1PodCondition]
            https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1PodCondition.md

    """
    if conds is None:
        return None

    maxDate = None
    maxCond = None
    for condition in conds:
        if maxDate is None:
            maxCond = condition
            maxDate = condition.last_transition_time
            continue

        if condition.last_transition_time > maxDate and condition.status == "True":
            maxCond = condition
            maxDate = condition.last_transition_time

    return {
        "message": maxCond.message,
        "reason": maxCond.reason,
        "status": maxCond.status,
        "type": maxCond.type}


common_pod_paths = [
    ['name', 'metadata.labels.name'],
    ['pod_name', 'metadata.name'],
    ['svc_name', 'metadata.labels.svc_name'],
    ['pod_status', 'status.phase'],
    ['creation_date', 'metadata.creation_timestamp',
        lambda x: x.strftime('%D')],
    ['token', 'metadata.labels.jupyter_token'],
    ['container_status', 'status.container_statuses',
        lambda x: read_containers_status(x)],
    ['condition', 'status.conditions', lambda x: read_conditions(x)]
]

pod_paths = [
    *common_pod_paths,
    ['svc_status', 'metadata.labels.svc_name', lambda x: read_svc_status(x)],
]

pod_paths_post = [
    *common_pod_paths,
    ['svc_status', 'metadata.labels.svc_name', lambda _: 'Running'],
]


# Routes #


# TODO: Simply checking ws.close isn't enough
# https://github.com/heroku-python/flask-sockets/issues/60
# Solution may be move to aiohttp
@sockets.route('/jupyter/ws')
def echo_socket(ws):
    user_id = ws.environ['HTTP_USER']
    w = kube.watch.Watch()

    for event in w.stream(k8s.list_namespaced_pod, namespace='default',
                          label_selector=f"user_id={UNSAFE_user_id_transform(user_id)}"):

        if ws.closed:
            log.info("Websocket closed")
            w.stop()
            return

        try:
            obj = event["object"]

            if event['type'] == 'MODIFIED' or event['type'] == 'ADDED':
                ws.send(
                    f'{{"event": "{event["type"]}", "resource": {marshall_json([obj], pod_paths, True)}}}')

            if event['type'] == 'DELETED':
                ws.send(
                    f'{{"event": "DELETED", "resource": {marshall_json([obj], pod_paths, True)}}}')

        except Exception as e:
            log.info("Issue with watcher")
            log.error(e)
            w.stop()
            break

    ws.close()


@app.route('/jupyter/verify/<svc_name>/', methods=['GET'])
def verify(svc_name: str):
    access_token = request.cookies.get('access_token')

    if is_falsy_str(access_token):
        return '', 401

    resp = requests.get(f'{AUTH_GATEWAY}/verify',
                        headers={'Authorization': f'Bearer {access_token}'})

    if resp.status_code != 200:
        return '', 401

    user_id = resp.headers.get('User')

    if is_falsy_str(user_id):
        return '', 401

    k_res = k8s.read_namespaced_service(svc_name, 'default')
    labels = k_res.metadata.labels

    if labels['user_id'] != UNSAFE_user_id_transform(user_id):
        return '', 401

    resp = Response('')
    resp.headers['IP'] = k_res.spec.cluster_ip

    return resp


@app.route('/jupyter', methods=['GET'])
def get_notebooks():
    user_id = request.headers.get('User')

    if is_falsy_str(user_id):
        return forbidden()

    # Not well-documented: empty selector returns nothing
    # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md
    # https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
    pods = k8s.list_namespaced_pod(
        namespace='default',
        label_selector=f"user_id={UNSAFE_user_id_transform(user_id)}",
        timeout_seconds=KUBERNETES_TIMEOUT_IN_SECONDS).items

    return marshall_json(pods, pod_paths), 200


@app.route('/jupyter/<svc_name>', methods=['DELETE'])
def delete_notebook(svc_name: str):
    # TODO: Is it possible to have a falsy token value and get here?
    user_id = request.headers.get("User")

    if is_falsy_str(user_id) or is_falsy_str(svc_name):
        return forbidden()

    escp_user_id = UNSAFE_user_id_transform(request.headers.get("User"))

    try:
        svc = k8s.read_namespaced_service(svc_name, 'default')

        if svc.metadata.labels['user_id'] != escp_user_id:
            return forbidden()

        del_svc(svc_name)

        pods = k8s.list_namespaced_pod(
            namespace='default',
            label_selector=f"uuid={svc.metadata.labels['uuid']}",
            timeout_seconds=KUBERNETES_TIMEOUT_IN_SECONDS).items

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

    except kube.client.rest.ApiException:
        # FIXME: Return non-200 errors, not done now because web app
        # needs to be updated to catch and handle 404 as a non-fatal response
        # return '', str(e)[1:4]
        return '', 200


@app.route('/jupyter', methods=['POST'])
def new_notebook():
    name = request.form.get('name', 'a_notebook')
    image = request.form.get('image', DEFAULT_HAIL_IMAGE)

    user_id = request.headers.get('User')

    if is_falsy_str(user_id) or image not in WORKER_IMAGES:
        return forbidden()

    # Token does not need to be cryptographically secure
    # Is trusted, and authorization is handled upstream
    jupyter_token = uuid.uuid4().hex
    _, pod = start_pod(
        jupyter_token, WORKER_IMAGES[image],
        labels={'name': name,
                'jupyter_token': jupyter_token,
                'user_id': UNSAFE_user_id_transform(user_id)}
    )

    # We could also construct pod_paths_post here, and close over svc
    return marshall_json([pod], pod_paths_post, True), 200


if __name__ == '__main__':
    server = pywsgi.WSGIServer(
        ('', 5000), app, handler_class=WebSocketHandler, log=log)

    server.serve_forever()
