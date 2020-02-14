import logging
import aiohttp
from aiohttp import web
import aiohttp_session
import uvloop
import google.auth.transport.requests
import google.oauth2.id_token
import google.cloud.storage
import google_auth_oauthlib.flow
from hailtop.config import get_deploy_config
from gear import setup_aiohttp_session, create_database_pool, \
    rest_authenticated_users_only, web_authenticated_developers_only, \
    web_maybe_authenticated_user, create_session, check_csrf_token
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, \
    set_message, render_template

log = logging.getLogger('auth')

uvloop.install()

deploy_config = get_deploy_config()

routes = web.RouteTableDef()


def get_flow(redirect_uri, state=None):
    scopes = [
        'https://www.googleapis.com/auth/userinfo.profile',
        'https://www.googleapis.com/auth/userinfo.email',
        'openid'
    ]
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        '/auth-oauth2-client-secret/client_secret.json', scopes=scopes, state=state)
    flow.redirect_uri = redirect_uri
    return flow


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('')
@routes.get('/')
async def get_index(request):  # pylint: disable=unused-argument
    return aiohttp.web.HTTPFound(deploy_config.external_url('auth', '/login'))


@routes.get('/login')
@web_maybe_authenticated_user
async def login(request, userdata):
    next = request.query.get('next', deploy_config.external_url('notebook', ''))
    if userdata:
        return aiohttp.web.HTTPFound(next)

    flow = get_flow(deploy_config.external_url('auth', '/oauth2callback'))

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    session = await aiohttp_session.new_session(request)
    session['state'] = state
    session['next'] = next

    return aiohttp.web.HTTPFound(authorization_url)


@routes.get('/oauth2callback')
async def callback(request):
    session = await aiohttp_session.get_session(request)
    if 'state' not in session:
        raise web.HTTPUnauthorized()
    state = session['state']

    flow = get_flow(deploy_config.external_url('auth', '/oauth2callback'), state=state)

    try:
        flow.fetch_token(code=request.query['code'])
        token = google.oauth2.id_token.verify_oauth2_token(
            flow.credentials.id_token, google.auth.transport.requests.Request())
        email = token['email']
    except Exception:
        log.exception('oauth2 callback: could not fetch and verify token')
        raise web.HTTPUnauthorized()

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT * FROM users WHERE email = %s AND state = 'active';", email)
            users = await cursor.fetchall()

    if len(users) != 1:
        raise web.HTTPUnauthorized()
    user = users[0]

    session_id = await create_session(dbpool, user['id'])

    del session['state']
    session['session_id'] = session_id
    next = session.pop('next')
    return aiohttp.web.HTTPFound(next)


@routes.post('/logout')
@check_csrf_token
@web_maybe_authenticated_user
async def logout(request, userdata):
    if not userdata:
        return web.HTTPFound(deploy_config.external_url('notebook', ''))

    dbpool = request.app['dbpool']
    session_id = userdata['session_id']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('DELETE FROM sessions WHERE session_id = %s;', session_id)

    session = await aiohttp_session.get_session(request)
    if 'session_id' in session:
        del session['session_id']

    return web.HTTPFound(deploy_config.external_url('notebook', ''))


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


@routes.get('/users')
@web_authenticated_developers_only()
async def get_users(request, userdata):
    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT * FROM users;')
            users = await cursor.fetchall()
    page_context = {
        'users': users
    }
    return await render_template('auth', request, userdata, 'users.html', page_context)


@routes.post('/users')
@check_csrf_token
@web_authenticated_developers_only()
async def post_create_user(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    dbpool = request.app['dbpool']
    post = await request.post()
    username = post['username']
    email = post['email']
    is_developer = post.get('is_developer') == '1'

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                '''
INSERT INTO users (state, username, email, is_developer)
VALUES (%s, %s, %s, %s);
''',
                ('creating', username, email, is_developer))
            user_id = cursor.lastrowid

    set_message(session, f'Created user {user_id} {username}.', 'info')

    return web.HTTPFound(deploy_config.external_url('auth', '/users'))


@routes.post('/users/delete')
@check_csrf_token
@web_authenticated_developers_only()
async def delete_user(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    dbpool = request.app['dbpool']
    post = await request.post()
    id = post['id']
    username = post['username']

    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            n_rows = await cursor.execute(
                '''
UPDATE users
SET state = 'deleting'
WHERE id = %s AND username = %s;
''',
                (id, username))
            if n_rows != 1:
                assert n_rows == 0
                set_message(session, f'Delete failed, no such user {id} {username}.', 'error')

    set_message(session, f'Deleted user {id} {username}.', 'info')

    return web.HTTPFound(deploy_config.external_url('auth', '/users'))


@routes.get('/api/v1alpha/oauth2callback')
async def rest_callback(request):
    state = request.query['state']
    code = request.query['code']
    callback_port = request.query['callback_port']

    try:
        flow = get_flow(f'http://127.0.0.1:{callback_port}/oauth2callback', state=state)
        flow.fetch_token(code=code)
        token = google.oauth2.id_token.verify_oauth2_token(
            flow.credentials.id_token, google.auth.transport.requests.Request())
        email = token['email']
    except Exception:
        log.exception('fetching and decoding token')
        raise web.HTTPUnauthorized()

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT * FROM users WHERE email = %s AND state = 'active';", email)
            users = await cursor.fetchall()

    if len(users) != 1:
        raise web.HTTPUnauthorized()
    user = users[0]

    session_id = await create_session(dbpool, user['id'])

    return web.json_response({
        'token': session_id,
        'username': user['username']
    })


@routes.post('/api/v1alpha/logout')
@rest_authenticated_users_only
async def rest_logout(request, userdata):
    session_id = userdata['session_id']
    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('DELETE FROM sessions WHERE session_id = %s;', session_id)

    return web.Response(status=200)


@routes.get('/api/v1alpha/userinfo')
async def userinfo(request):
    if 'Authorization' not in request.headers:
        log.info('Authorization not in request.headers')
        raise web.HTTPUnauthorized()

    auth_header = request.headers['Authorization']
    if not auth_header.startswith('Bearer '):
        log.info('Bearer not in Authorization header')
        raise web.HTTPUnauthorized()
    session_id = auth_header[7:]

    # b64 encoding of 32-byte session ID is 44 bytes
    if len(session_id) != 44:
        log.info('Session id != 44 bytes')
        raise web.HTTPUnauthorized()

    dbpool = request.app['dbpool']
    async with dbpool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
SELECT users.*, sessions.session_id FROM users
INNER JOIN sessions ON users.id = sessions.user_id
WHERE users.state = 'active' AND (sessions.session_id = %s) AND (ISNULL(sessions.max_age_secs) OR (NOW() < TIMESTAMPADD(SECOND, sessions.max_age_secs, sessions.created)));
''', session_id)
            users = await cursor.fetchall()

    if len(users) != 1:
        log.info(f'Unknown session id: {session_id}')
        raise web.HTTPUnauthorized()
    user = users[0]

    return web.json_response(user)


async def on_startup(app):
    app['dbpool'] = await create_database_pool()


async def on_cleanup(app):
    dbpool = app['dbpool']
    dbpool.close()
    await dbpool.wait_closed()


def run():
    app = web.Application()

    setup_aiohttp_jinja2(app, 'auth')
    setup_aiohttp_session(app)

    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(deploy_config.prefix_application(app, 'auth'), host='0.0.0.0', port=5000)
