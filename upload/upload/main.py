# based on: https://developers.google.com/identity/protocols/OAuth2WebServer

# useful links:
# https://developers.google.com/api-client-library/python/
# https://developers.google.com/api-client-library/python/apis/
# https://developers.google.com/api-client-library/python/start/get_started
# https://developers.google.com/identity/protocols/googlescopes

import logging
import re
import datetime
import flask
from flask import Flask, Response, session, request, redirect, render_template, escape, jsonify, abort
import httplib2
import json
import requests

import google.oauth2.credentials
import google_auth_oauthlib.flow

from apiclient.discovery import build

fmt = logging.Formatter(
  # NB: no space after levelname because WARNING is so long
  '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
  '%(message)s')

fh = logging.FileHandler('upload.log')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)

log = logging.getLogger('upload')
log.setLevel(logging.INFO)

logging.basicConfig(
  handlers=[fh, ch],
  level=logging.INFO)

USERS = set([
    'cseed@broadinstitute.org',
    'dking@broadinstitute.org',
    'jbloom@broadinstitute.org',
    'jigold@broadinstitute.org',
    'cvittal@broadinstitute.org',
    'tpoterba@broadinstitute.org',
    'pschultz@broadinstitute.org',
    'wang@broadinstitute.org'
])

CLIENT_SECRET = '/upload-oauth2-client-secret/client_secret.json'
SCOPES = [
  'https://www.googleapis.com/auth/userinfo.email',
  'https://www.googleapis.com/auth/userinfo.profile',
  'https://www.googleapis.com/auth/plus.me'
]

counter = 0
def next_id():
    global counter

    counter = counter + 1
    return counter

def url_for(*args, **kwargs):
    # NOTE: nginx strips https and sets X-Forwarded-Proto: https, but
    # it is not used by request.url or url_for, so rewrite the url and
    # set _scheme='https' explicitly.
    kwargs['_scheme'] = 'https'
    return flask.url_for(*args, **kwargs)

id_item = {}
items = []

class Item(object):
    def __init__(self, data):
        self.id = next_id()
        
        items.insert(0, self)
        id_item[self.id] = self

        self.timestamp = datetime.datetime.now()
        self.data = data

    def typ(self):
        return self.data['type']

    def email(self):
        return self.data['email']

    def config(self):
        return self.data['config']

    def hail_version(self):
        return self.config()['hail_version']
    
    def contents(self):
        return self.data['contents']

    def preview(self):
        s = self.contents()
        if len(s) > 40:
            s = s[:40] + " ..."
        return re.sub('\\s+', ' ', s)

app = Flask('upload')

with open('/flask-secret-key/flask-secret-key', 'rb') as f:
    app.secret_key = f.read()

def get_flow(state=None):
  flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
    CLIENT_SECRET, scopes=SCOPES, state=state)

  flow.redirect_uri = url_for('oauth2callback', _external=True)
  
  return flow

@app.route('/', methods=['GET'])
def index():
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']
    
    # just show 50
    page_items = items[:50]
    
    return render_template('index.html', email=email, page_items=page_items, n_items=len(items))

@app.route('/login', methods=['GET'])
def login():
  flow = get_flow()
  authorization_url, state = flow.authorization_url(
    access_type='offline',
    include_granted_scopes='true')

  session['state'] = state

  return redirect(authorization_url)

@app.route('/oauth2callback', methods=['GET'])
def oauth2callback():
  authorization_response = request.url
  # see comment in url_for
  authorization_response = re.sub('^http://', 'https://', authorization_response)
  
  flow = get_flow(state=session['state'])
  flow.fetch_token(authorization_response=authorization_response)
  credentials = flow.credentials
  
  oauth2v2 = build('oauth2', 'v2', credentials=credentials)
  profile = oauth2v2.userinfo().v2().me().get().execute()
  email = profile['email']
  
  if email in USERS:
    session['credentials'] = {
      'token': credentials.token,
      'refresh_token': credentials.refresh_token,
      'token_uri': credentials.token_uri,
      'client_id': credentials.client_id,
      'client_secret': credentials.client_secret,
      'scopes': credentials.scopes
    }
    session['email'] = email
    
    return redirect(url_for('index'))
  else:
    abort(401)

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    Item(data)
    return jsonify({})

@app.route('/items/<int:item_id>/contents', methods=['GET'])
def get_item_contents(item_id):
    if 'email' not in session:
        return redirect(url_for('login'))

    item = id_item.get(item_id)
    if not item:
        abort(404)
    return Response(item.contents(), mimetype='text/plain')

@app.route('/items/<int:item_id>/config', methods=['GET'])
def get_item_config(item_id):
    if 'email' not in session:
        return redirect(url_for('login'))

    item = id_item.get(item_id)
    if not item:
        abort(404)
    return Response(json.dumps(item.config(), indent=2), mimetype='text/plain')

@app.route('/logout', methods=['GET'])
def logout():
    credentials = session.get('credentials')
    if credentials:
        requests.post('https://accounts.google.com/o/oauth2/revoke',
                      params={'token': credentials.token},
                      headers={'content-type': 'application/x-www-form-urlencoded'})
    session.pop('credentials', None)
    session.pop('email', None)
    return redirect(url_for('index'))

app.run(threaded=False, host='0.0.0.0')
