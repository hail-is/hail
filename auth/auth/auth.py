import json
import logging
import datetime
import base64
import secrets
import aiohttp
from aiohttp import web
import aiohttp_session
import aiohttp_session.cookie_storage
import aiomysql
import uvloop

import google.auth.transport.requests
import google.oauth2.id_token
import google_auth_oauthlib.flow

from hailtop.gear import configure_logging, setup_aiohttp_session
from hailtop.gear.auth import get_jwtclient, rest_get_session_id

log = logging.getLogger('auth')

uvloop.install()

routes = web.RouteTableDef()

CLIENT_SECRET = '/auth-oauth2-client-secret/client_secret.json'
SCOPES = [
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
    'openid'
]


def get_flow(redirect_uri, state=None):
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRET, scopes=SCOPES, state=state)
    flow.redirect_uri = redirect_uri
    return flow


@routes.get('/')
async def get_index(request):  # pylint: disable=unused-argument
    return aiohttp.web.HTTPFound('/login')


@routes.get('/login')
async def login(request):
    flow = get_flow('https://auth.hail.is/oauth2callback')

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

    flow = get_flow('https://auth.hail.is/oauth2callback', state=state)

    try:
        flow.fetch_token(code=request.query['code'])
        token = google.oauth2.id_token.verify_oauth2_token(
            flow.credentials.id_token, google.auth.transport.requests.Request())
        log.info(f'token {token}')
        id = token['sub']
    except Exception:
        log.exception('oauth2 callback: could not fetch and verify token')
        raise unauth

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * from users.user_data where user_id = %s;', f'google-oauth2|{id}')
            users = await cursor.fetchall()
            log.info(f'users {len(users)} {users}')

    if len(users) != 1:
        raise unauth
    user = users[0]

    session_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii')

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO users.sessions (session_id, kind, user_id, max_age_secs) VALUES (%s, %s, %s, %s);',
                                 # 2592000s = 30d
                                 (session_id, 'web', user['id'], 2592000))

    del session['state']
    next = session.pop('next', 'https://notebook2.hail.is')
    session['session_id'] = session_id
    return aiohttp.web.HTTPFound(next)


@routes.post('/logout')
async def logout(request):
    session = await aiohttp_session.get_session(request)
    session_id = session.pop('session_id', None)
    if session_id:
        dbpool = request.app['dbpool']
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('DELETE FROM user.sessions WHERE session_id = %s;', session_id)

    session.invalidate()

    # FIXME
    return web.Response(status=200)


@routes.get('/api/v1alpha/login')
async def rest_login(request):
    callback_port = request.query['callback_port']

    flow = get_flow(f'http://127.0.0.1:{callback_port}/oauth2callback')
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    return web.json_response({
        'authorization_url': authorization_url,
        'state': state
    })


@routes.get('/api/v1alpha/oauth2callback')
async def rest_callback(request):
    unauth = web.HTTPUnauthorized(headers={'WWW-Authenticate': 'Bearer'})

    state = request.query['state']
    code = request.query['code']
    callback_port = request.query['callback_port']

    try:
        flow = get_flow(f'http://127.0.0.1:{callback_port}/oauth2callback', state=state)
        flow.fetch_token(code=code)
        token = google.oauth2.id_token.verify_oauth2_token(
            flow.credentials.id_token, google.auth.transport.requests.Request())
        id = token['sub']
    except Exception:
        log.exception('fetching and decoding token')
        raise unauth

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * from users.user_data where user_id = %s;', f'google-oauth2|{id}')
            users = await cursor.fetchall()
            log.info(f'users {len(users)} {users}')

    if len(users) != 1:
        raise unauth
    user = users[0]

    session_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii')

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('INSERT INTO users.sessions (session_id, kind, user_id) VALUES (%s, %s, %s, %s);',
                                 # 2592000s = 30d
                                 (session_id, 'web', user['id'], 2592000))

    token = get_jwtclient().encode({
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30),
        'sub': session_id
    })

    return web.json_response({
        'token': token,
        'username': user['username']
    })


@routes.post('/api/v1alpha/logout')
async def rest_logout(request):
    session_id = await rest_get_session_id(request)
    if session_id:
        dbpool = request.app['dbpool']
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('DELETE FROM user.sessions WHERE session_id = %s;', session_id)

    return web.Response(status=200)


@routes.post('/api/v1alpha/userinfo')
async def rest_userinfo(request):
    unauth = web.HTTPUnauthorized(headers={'WWW-Authenticate': 'token'})

    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return unauth
    if not auth_header.startswith('token '):
        return unauth
    session_id = auth_header[6:]

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
SELECT user_data.* FROM user_data
INNER JOIN sessions ON user_data.id = sessions.user_id
WHERE (TIMESTAMPADD(SECOND, sessions.max_age_secs, sessions.created) < NOW()) AND sessions.session_id = %s)
''', session_id)
            users = await cursor.fetchall()

    if len(users) != 1:
        raise unauth
    user = users[0]

    return web.json_response(user)


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
    setup_aiohttp_session(app)
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host='0.0.0.0', port=5000)
