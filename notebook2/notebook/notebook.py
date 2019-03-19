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
import jwt
import base64
import mysql.connector

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

SQL_HOST_DEF = os.environ.get('SQL_HOST')

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
                    f'--NotebookApp.base_url=/instance/{pod_id}/',
                    "--ip", "0.0.0.0", "--no-browser"
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
        return {"running": {"started_at": state.running.started_at}}

    if state.waiting:
        return {"waiting": {"reason": state.waiting.reason}}

    if state.terminated:
        return {"terminated": {
                                "exit_code": state.terminated.exit_code,
                                "finished_at": state.terminated.finished_at,
                                "started_at": state.terminated.started_at,
                                "reason": state.terminated.reason
                              }
                }


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


class Table:
    @staticmethod
    def getSecret(b64str):
        return base64.b64decode(b64str).decode('utf-8')

    @staticmethod
    def getSecrets():
        secrets = {}

        try:
            res = k8s.read_namespaced_secret('user-secrets', 'default')
            data = res.data

            if SQL_HOST_DEF is not None:
                host = SQL_HOST_DEF
            else:
                host = Table.getSecret(data['host'])

            secrets['user'] = Table.getSecret(data['user'])
            secrets['password'] = Table.getSecret(data['password'])
            secrets['db'] = Table.getSecret(data['db'])
            secrets['host'] = host
        except kube.client.rest.ApiException as e:
            print(e)

        return secrets

    def __init__(self):
        secrets = Table.getSecrets()

        if not secrets:
            raise "Couldn't read user-secrets"

        self.cnx = mysql.connector.connect(**secrets)
        cursor = self.cnx.cursor()

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS users (
                    id INT NOT NULL AUTO_INCREMENT,
                    user_id VARCHAR(255) NOT NULL,
                    gsa_name VARCHAR(255) NOT NULL,
                    ksa_name VARCHAR(255) NOT NULL,
                    bucket_name VARCHAR(255) NOT NULL,
                    PRIMARY KEY (id),
                    UNIQUE INDEX auth0_id (user_id)
                ) ENGINE=INNODB;
            """
        )
        self.cnx.commit()
        cursor.close()

    def get(self, user_id):
        cursor = self.cnx.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
        res = cursor.fetchone()

        cursor.close()

        return res

    def insert(self, user_id, gsa_name, ksa_name, bucket_name):
        cursor = self.cnx.cursor()
        cursor.execute(
            """
            INSERT INTO users
                (user_id, gsa_name, ksa_name, bucket_name)
                VALUES (%s, %s, %s, %s)
            """, (user_id, gsa_name, ksa_name, bucket_name))
        self.cnx.commit()

        cnt = cursor.rowcount

        cursor.close()

        return cnt == 1


user_table = Table()

@app.route('/healthcheck')
def healthcheck():
    return '', 200


@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')


@app.route('/notebook', methods=['GET'])
@requires_auth()
def notebook_page():
    notebooks = get_live_user_notebooks(user_id = user_id_transform(session['user']['auth0_id']))

    # https://github.com/hail-is/hail/issues/5487
    assert len(notebooks) <= 1

    if len(notebooks) == 0:
        return render_template('notebook.html',
                               form_action_url=external_url_for('notebook'),
                               images=list(WORKER_IMAGES),
                               default='hail-jupyter')

    session['notebook'] = notebooks[0]

    return render_template('notebook.html', notebook=notebooks[0])


@app.route('/notebook/delete', methods=['POST'])
@requires_auth()
def notebook_delete():
    notebook = session.get('notebook')

    if notebook is not None:
        delete_worker_pod(notebook['pod_name'])
        del session['notebook']

    return redirect(external_url_for('notebook'))


@app.route('/notebook', methods=['POST'])
@requires_auth()
def notebook_post():
    image = request.form['image']

    if image not in WORKER_IMAGES:
        return '', 404

    jupyter_token = uuid.uuid4().hex
    name = request.form.get('name', 'a_notebook')
    safe_user_id = user_id_transform(session['user']['auth0_id'])

    pod = start_pod(jupyter_token, WORKER_IMAGES[image], name, safe_user_id)
    session['notebook'] = notebooks_for_ui([pod])[0]

    return redirect(external_url_for('notebook'))


@app.route('/auth/<requested_pod_uuid>')
@requires_auth()
def auth(requested_pod_uuid):
    notebook = session.get('notebook')

    if notebook is not None and notebook['pod_uuid'] == requested_pod_uuid:
        res = flask.make_response()
        res.headers['pod_ip'] = f"{notebook['pod_ip']}:{POD_PORT}"
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
    notebook = session['notebook']

    if notebook is None:
        return

    pod_uuid = notebook['pod_uuid']
    url = external_url_for(f'instance-ready/{pod_uuid}/')

    attempts = -1
    while attempts < 10:
        attempts += 1
        # Protect against many requests being issued
        # This may happen if client re-reqeusts after seeing 405
        # which can occur if it relies on container status,
        # which may not be "Ready" immediately after jupyter server is reachable
        # https://youtu.be/dp-RYuH4tmw
        gevent.sleep(1)

        try:
            response = requests.head(url, timeout=1, cookies=request.cookies)

            if response.status_code == 502:
                log.info(f'Pod not reachable for pod_uuid: {pod_uuid} : response: {response}')
                continue

            if response.status_code == 405:
                log.info(f'HEAD on jupyter succeeded for pod_uuid: {pod_uuid} : response: {response}')
            else:
                log.info(f'HEAD on jupyter failed for pod_uuid: {pod_uuid} : response: {response}')

            break
        except requests.exceptions.Timeout:
            log.info(f'HEAD on jupyter timed out for pod_uuid : {pod_uuid}')

    ws.send("1")


@app.route('/auth0-callback')
def auth0_callback():
    auth0.authorize_access_token()

    userinfo = auth0.get('userinfo').json()

    email = userinfo['email']

    if AUTHORIZED_USERS.get(email) is None:
        return redirect(external_url_for(f"error?err=Unauthorized"))

    session['user'] = {
        'auth0_id': userinfo['sub'],
        'name': userinfo['name'],
        'email': email,
        'picture': userinfo['picture'],
    }

    if 'referrer' in session:
        referrer = session['referrer']
        del session['referrer']
        return redirect(referrer)

    secrets = user_table.get(userinfo['sub'])
    print("SECRETS", userinfo['sub'], secrets)

    res = flask.make_response()
    res.set_cookie('user', {
        'auth0_id': session['user']['auth0_id'],
        'name': session['user']['name'],
        'email': session['user']['email'],
        **secrets
    })

    return redirect('/')


@app.route('/error', methods=['GET'])
def error_page():
    return render_template('error.html', error = request.args.get('err'))


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login_auth0():
    workshop_password = request.form['workshop-password']

    if workshop_password != '':
        if workshop_password != PASSWORD:
            return redirect(external_url_for(f"error?err=Unauthorized"))

        session['user'] = {
            'id': uuid.uuid4().hex,
            'name': "Guest",
            'email': "N/A",
            'picture': "N/A",
        }

        if 'referrer' in session:
            referrer = session['referrer']
            del session['referrer']
            return redirect(referrer)

        return redirect(external_url_for(''))

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
