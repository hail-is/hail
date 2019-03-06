"""
A Jupyter notebook service with local-mode Hail pre-installed
"""

import gevent
# must happen before anytyhing else
from gevent import monkey; monkey.patch_all()

from flask import Flask, session, redirect, render_template, request
from flask_sockets import Sockets
import flask
import sass
from authlib.flask.client import OAuth
from urllib.parse import urlencode
from functools import wraps

import kubernetes as kube

import logging
import os
import re
import requests
import time
import uuid
import hashlib
import ujson

fmt = logging.Formatter(
   # NB: no space after levelname because WARNING is so long
   '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
   '%(message)s')

fh = logging.FileHandler('notebook2.log')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)

log = logging.getLogger('notebook2')
log.setLevel(logging.INFO)
logging.basicConfig(
    handlers=[fh, ch],
    level=logging.INFO)

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()

kube.config.connection_pool_maxsize = 5000
k8s = kube.client.CoreV1Api()

app = Flask(__name__)
sockets = Sockets(app)
oauth = OAuth(app)

scss_path = os.path.join(app.static_folder, 'styles')
css_path = os.path.join(app.static_folder, 'css')
os.makedirs(css_path, exist_ok=True)

sass.compile(dirname=(scss_path, css_path), output_style='compressed')

def read_string(f):
    with open(f, 'r') as f:
        return f.read().strip()

# Must be int for Kubernetes V1 api timeout_seconds property
KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5))

AUTHORIZED_USERS = read_string('/notebook-secrets/authorized-users').split(',')
AUTHORIZED_USERS = dict((email, True) for email in AUTHORIZED_USERS)

PASSWORD = read_string('/notebook-secrets/password')
ADMIN_PASSWORD = read_string('/notebook-secrets/admin-password')
INSTANCE_ID = uuid.uuid4().hex

POD_PORT = 8888

app.config.update(
    SECRET_KEY = read_string('/notebook-secrets/secret-key'),
    SESSION_COOKIE_SAMESITE = 'lax',
    SESSION_COOKIE_HTTPONLY = True,
    SESSION_COOKIE_SECURE = os.environ.get("NOTEBOOK_DEBUG") != "1"
)

AUTH0_CLIENT_ID = 'Ck5wxfo1BfBTVbusBeeBOXHp3a7Z6fvZ'
AUTH0_BASE_URL = 'https://hail.auth0.com'

auth0 = oauth.register(
    'auth0',
    client_id = AUTH0_CLIENT_ID,
    client_secret = read_string('/notebook-secrets/auth0-client-secret'),
    api_base_url = AUTH0_BASE_URL,
    access_token_url = f'{AUTH0_BASE_URL}/oauth/token',
    authorize_url = f'{AUTH0_BASE_URL}/authorize',
    client_kwargs = {
        'scope': 'openid email profile',
    },
)

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
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


def requires_auth(for_page = True):
    def auth(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user' not in session:
                if for_page:
                    session['referrer'] = request.url
                    return redirect(external_url_for('login'))

                return '', 401

            return f(*args, **kwargs)

        return decorated
    return auth


def start_pod(jupyter_token, image, name, user_id):
    pod_id = uuid.uuid4().hex

    pod_spec = kube.client.V1PodSpec(
        containers=[
            kube.client.V1Container(
                command=[
                    'jupyter',
                    'notebook',
                    f'--NotebookApp.token={jupyter_token}',
                    f'--NotebookApp.base_url=/instance/{pod_id}/'
                ],
                name='default',
                image=image,
                ports=[kube.client.V1ContainerPort(container_port=POD_PORT)],
                resources=kube.client.V1ResourceRequirements(
                    requests={'cpu': '1.601', 'memory': '1.601G'}),
                readiness_probe=kube.client.V1Probe(
                    period_seconds=5,
                    http_get=kube.client.V1HTTPGetAction(
                        path=f'/instance/{pod_id}/login',
                        port=POD_PORT)))])
    pod_template = kube.client.V1Pod(
        metadata=kube.client.V1ObjectMeta(
            generate_name='notebook2-worker-',
            labels={
                'app': 'notebook2-worker',
                'hail.is/notebook2-instance': INSTANCE_ID,
                'uuid': pod_id,
                'name': name,
                'jupyter_token': jupyter_token,
                'user_id': user_id
            }),
        spec=pod_spec)
    pod = k8s.create_namespaced_pod(
        'default',
        pod_template,
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)

    return pod


def external_url_for(path):
    # NOTE: nginx strips https and sets X-Forwarded-Proto: https, but
    # it is not used by request.url or url_for, so rewrite the url and
    # set _scheme='https' explicitly.
    protocol = request.headers.get('X-Forwarded-Proto', None)
    url = flask.url_for('root', _scheme=protocol, _external='true')
    return url + path


# Kube has max 63 character limit
def user_id_transform(user_id): return hashlib.sha224(user_id.encode('utf-8')).hexdigest()


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

    assert(len(container_statuses) == 1)

    state = container_statuses[0].state

    if state.running:
        return {"running": {}}

    if state.waiting:
        return {"waiting": {"reason": state.waiting.reason}}

    if state.terminated:
        return {"terminated": {}}


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
        'name': pod.metadata.labels['name'],
        'pod_name': pod.metadata.name,
        'pod_status': pod.status.phase,
        'pod_uuid': pod.metadata.labels['uuid'],
        'pod_ip': pod.status.pod_ip,
        'creation_date': pod.metadata.creation_timestamp.strftime('%D'),
        'jupyter_token': pod.metadata.labels['jupyter_token'],
        'container_status': container_status_for_ui(pod.status.container_statuses),
        'condition': pod_condition_for_ui(pod.status.conditions),
        'deletion_timestamp': pod.metadata.deletion_timestamp
    }

    notebook['url'] = f"/instance/{notebook['pod_uuid']}/?token={notebook['jupyter_token']}"

    return notebook


def notebooks_for_ui(pods):
    return [pod_to_ui_dict(pod) for pod in pods]


def get_live_user_notebooks(user_id):
    pods = k8s.list_namespaced_pod(
        namespace='default',
        label_selector=f"user_id={user_id}",
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS).items

    return list(filter(lambda n: n['deletion_timestamp'] is None, notebooks_for_ui(pods)))


def notebook_to_dict(notebooks):
    return dict((n['pod_uuid'], n) for n in notebooks)


@app.route('/healthcheck')
def healthcheck():
    return '', 200


@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')


@app.route('/notebook', methods=['GET'])
@requires_auth()
def notebook_page():
    notebooks = get_live_user_notebooks(user_id = user_id_transform(session['user']['id']))

    if len(notebooks) == 0:
        return render_template('notebook.html',
                               form_action_url=external_url_for('notebook'),
                               images=list(WORKER_IMAGES),
                               default='hail')

    session['notebooks'] = notebook_to_dict(notebooks)

    return render_template('notebook.html', notebooks=notebooks)


@app.route('/notebook/<pod_uuid>/delete', methods=['POST'])
@requires_auth()
def notebook_delete(pod_uuid):
    notebooks = session.get('notebooks')

    if notebooks is not None and pod_uuid in notebooks:
        delete_worker_pod(notebooks[pod_uuid]['pod_name'])
        del session['notebooks'][pod_uuid]

    return redirect(external_url_for('notebook'))


@app.route('/notebook', methods=['POST'])
@requires_auth()
def notebook_post():
    image = request.form['image']

    if image not in WORKER_IMAGES:
        return '', 404

    jupyter_token = uuid.uuid4().hex
    name = request.form.get('name', 'a_notebook')
    safe_user_id = user_id_transform(session['user']['id'])

    pod = start_pod(jupyter_token, WORKER_IMAGES[image], name, safe_user_id)
    session['notebooks'] = notebook_to_dict(notebooks_for_ui([pod]))

    return redirect(external_url_for('notebook'))


@app.route('/auth/<requested_pod_uuid>')
@requires_auth()
def auth(requested_pod_uuid):
    notebooks = session.get('notebooks')

    if notebooks is not None and requested_pod_uuid in notebooks:
        res = flask.make_response()
        res.headers['pod_ip'] = f"{notebooks[requested_pod_uuid]['pod_ip']}:{POD_PORT}"
        return res

    return '', 404


def get_all_workers():
    return k8s.list_namespaced_pod(
        namespace='default',
        watch=False,
        label_selector='app=notebook2-worker',
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)


@app.route('/workers')
@requires_auth()
def workers():
    if not session.get('admin'):
        return redirect(external_url_for('admin-login'))

    return render_template('workers.html',
                           workers=get_all_workers(),
                           workers_url=external_url_for('workers'),
                           leader_instance=INSTANCE_ID)


@app.route('/workers/<pod_name>/delete')
@requires_auth()
def workers_delete(pod_name):
    if not session.get('admin'):
        return redirect(external_url_for('admin-login'))

    delete_worker_pod(pod_name)

    return redirect(external_url_for('workers'))


@app.route('/workers/delete-all-workers', methods=['POST'])
@requires_auth()
def delete_all_workers():
    if not session.get('admin'):
        return redirect(external_url_for('admin-login'))

    for pod_name in get_all_workers():
        delete_worker_pod(pod_name)

    return redirect(external_url_for('workers'))


def delete_worker_pod(pod_name):
    try:
        k8s.delete_namespaced_pod(
            pod_name,
            'default',
            kube.client.V1DeleteOptions(),
            _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
    except kube.client.rest.ApiException as e:
        log.info(f'pod {pod_name} already deleted {e}')


@app.route('/admin-login', methods=['GET'])
@requires_auth()
def admin_login():
    return render_template('admin-login.html',
                           form_action_url=external_url_for('admin-login'))


@app.route('/admin-login', methods=['POST'])
@requires_auth()
def admin_login_post():
    if request.form['password'] != ADMIN_PASSWORD:
        return '403 Forbidden', 403

    session['admin'] = True

    return redirect(external_url_for('workers'))


@app.route('/worker-image')
@requires_auth()
def worker_image():
    return '\n'.join(WORKER_IMAGES.values()), 200


@sockets.route('/wait')
@requires_auth(for_page = False)
def wait_websocket(ws):
    kube_safe_user_id = user_id_transform(session['user']['id'])

    w = kube.watch.Watch()
    for event in w.stream(k8s.list_namespaced_pod, namespace='default',
                          label_selector=f"app=notebook2-worker,user_id={kube_safe_user_id}"):

        try:
            pod = event["object"]

            ws.send(
                    f'{{"event": "{event["type"]}", "resource_id": "{pod.metadata.name}", "resource": {ujson.dumps(notebooks_for_ui([pod]))}}}')

        except Exception as e:
            log.error(e)
            break

    w.stop()


@app.route('/auth0-callback')
def auth0_callback():
    auth0.authorize_access_token()

    userinfo = auth0.get('userinfo').json()

    email = userinfo['email']
    workshop_password = session['workshop_password']
    del session['workshop_password']

    if AUTHORIZED_USERS.get(email) is None and workshop_password != PASSWORD:
        return redirect(external_url_for(f"error?err=Unauthorized"))

    session['user'] = {
        'id': userinfo['sub'],
        'name': userinfo['name'],
        'email': email,
        'picture': userinfo['picture'],
    }

    if 'referrer' in session:
        referrer = session['referrer']
        del session['referrer']
        return redirect(referrer)

    return redirect('/')


@app.route('/error', methods=['GET'])
def error_page():
    return render_template('error.html', error = request.args.get('err'))


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login_auth0():
    session['workshop_password'] = request.form.get('workshop-password')

    return auth0.authorize_redirect(redirect_uri = external_url_for('auth0-callback'),
                                    audience = f'{AUTH0_BASE_URL}/userinfo', prompt = 'login')


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    params = {'returnTo': external_url_for(''), 'client_id': AUTH0_CLIENT_ID}
    return redirect(auth0.api_base_url + '/v2/logout?' + urlencode(params))


if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler, log=log)
    server.serve_forever()
