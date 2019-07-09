import re
import json
import logging
import asyncio
import datetime
from cryptography import fernet
import aiohttp
from aiohttp import web
import aiohttp_session
import aiohttp_session.cookie_storage
import aiomysql
import uvloop
import base64
import secrets

import google.auth.transport.requests
import google.oauth2.id_token
import google_auth_oauthlib.flow

from hailtop.gear import configure_logging

log = logging.getLogger('auth')

uvloop.install()

routes = web.RouteTableDef()

CLIENT_SECRET = '/auth-oauth2-client-secret/client_secret.json'
SCOPES = [
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
    'openid'
]


def get_flow(state=None):
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
      CLIENT_SECRET, scopes=SCOPES, state=state)
    flow.redirect_uri = 'https://auth.hail.is/oauth2callback'
    return flow


@routes.get('/')
async def get_index(request):
    return aiohttp.web.HTTPFound('/login')


@routes.get('/login')
async def login(request):
    flow = get_flow()

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    session = await aiohttp_session.get_session(request)
    session['state'] = state
    next = request.query.get('next')
    if next:
        session['next'] = next
    elif 'next' in session:
        del session['next']

    return aiohttp.web.HTTPFound(authorization_url)


@routes.get('/oauth2callback')
async def callback(request):
    unauth = web.HTTPUnauthorized(headers={'WWW-Authenticate': 'Bearer'})

    session = await aiohttp_session.get_session(request)
    if 'state' not in session:
      raise unauth
    state = session['state']

    flow = get_flow(state=state)
    authorization_response = re.sub('^http://', 'https://', str(request.url))

    try:
        # FIXME switch to code
        flow.fetch_token(authorization_response=authorization_response)
        token = google.oauth2.id_token.verify_oauth2_token(
          flow.credentials.id_token, google.auth.transport.requests.Request())
        id = token['sub']
    except Exception as e:
      log.exception('oauth2 callback: could not fetch and verify token')
      raise unauth

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * from users.user_data where id = %s;', f'google-oauth2|{id}')
            users = cursor.fetchall()

    if len(users) != 1:
      raise unauth
    user = users[0]

    session_id = secrets.token_bytes(32)

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO users.sessions (session_id, kind, user_id) VALUES (%s, %s, %s);',
                                 session_id, 'web', user['id'])

    del session['state']
    next = session.pop('next', 'https://notebook2.hail.is')
    session['session_id'] = base64.urlsafe_b64decode(session_id)
    return aiohttp.web.HTTPFound(next)


@routes.post('/logout')
async def logout():
    session = await aiohttp_session.get_session(request)
    session_id = session.pop('sesssion_id', None)
    if session_id:
        dbpool = request.app['dbpool']
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('DELETE FROM user.sessions WHERE session_id = %s;', session_id)

    session.invalidate()

    # FIXME
    return web.Response(status=200)


@routes.post('/log')
async def log():
    session = await aiohttp_session.get_session(request)
    log.info(f'log: aiohttp_session {session}')

    session_id = session.get('session_id')
    if session_id:
      dbpool = request.app['dbpool']
      async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
          await cursor.execute('SELECT * from users.sessions where session_id = %s;', session_id)
          sessions = cursor.fetchall()
          log.info(f'log: sessions {sessions}')

    return web.Response(status=200)


async def on_startup(app):
    with open('/sql-users-users-user-config/sql-config.json', 'r') as f:
        config = json.loads(f.read().strip())
        app['dbpool'] = await aiomysql.create_pool(host=config['host'],
                                                   port=config['port'],
                                                   db=config['db'],
                                                   user=config['user'],
                                                   password=config['password'],
                                                   charset='utf8',
                                                   cursorclass=aiomysql.cursors.DictCursor,
                                                   autocommit=True)


async def on_cleanup(app):
    dbpool = app['dbpool']
    dbpool.close()
    await dbpool.wait_closed()


def run():
    configure_logging()
    app = web.Application()
    with open('/aiohttp-session-secret-key/aiohttp-session-secret-key', 'rb') as f:
      fernet_key = f.read()
    secret_key = base64.urlsafe_b64decode(fernet_key)
    aiohttp_session.setup(app, aiohttp_session.cookie_storage.EncryptedCookieStorage(
      secret_key,
      cookie_name='session',
      max_age=30 * 24 * 60 * 60))
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host='0.0.0.0', port=5000)
