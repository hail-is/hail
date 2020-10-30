import logging
import asyncio
import aiohttp
from aiohttp import web
import aiohttp_session
import uvloop
import google.auth.transport.requests
import google.oauth2.id_token
import google_auth_oauthlib.flow
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.hail_logging import AccessLogger
from hailtop.utils import secret_alnum_string
from gear import (
    setup_aiohttp_session,
    rest_authenticated_users_only, web_authenticated_developers_only,
    web_maybe_authenticated_user, web_authenticated_users_only, create_session,
    check_csrf_token, transaction, Database
)
from web_common import (
    setup_aiohttp_jinja2, setup_common_static_routes, set_message,
    render_template
)

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


async def user_from_email(db, email):
    users = [x async for x in db.select_and_fetchall(
        "SELECT * FROM users WHERE email = %s;", email)]
    if len(users) == 1:
        return users[0]
    assert len(users) == 0, len(users)
    return None


def cleanup_session(session):
    def _delete(key):
        if key in session:
            del session[key]

    _delete('pending')
    _delete('email')
    _delete('state')
    _delete('next')
    _delete('caller')
    _delete('session_id')


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('')
@routes.get('/')
async def get_index(request):  # pylint: disable=unused-argument
    return aiohttp.web.HTTPFound(deploy_config.external_url('auth', '/login'))


@routes.get('/signup')
@web_maybe_authenticated_user
async def signup(request, userdata):
    db = request.app['db']

    next = request.query.get('next', deploy_config.external_url('notebook', ''))

    session = await aiohttp_session.get_session(request)

    if 'pending' in session:
        email = session['email']
        user = await user_from_email(db, email)

        if user is None:
            cleanup_session(session)
            return aiohttp.web.HTTPFound(next)

        if user['state'] == 'deleting' or user['state'] == 'deleted':
            cleanup_session(session)
            set_message(session, f'Account for {user["username"]} is deleted.', 'error')
            return aiohttp.web.HTTPFound(next)

        if user['state'] == 'active':
            session_id = await create_session(db, user['id'])
            cleanup_session(session)
            session['session_id'] = session_id
            return aiohttp.web.HTTPFound(next)

        assert user['state'] == 'creating'

        page_context = {
            'username': user['username'],
            'state': user['state'],
            'email': user['email']
        }

        return await render_template('auth', request, userdata, 'signup.html', page_context)

    flow = get_flow(deploy_config.external_url('auth', '/oauth2callback'))

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    session = await aiohttp_session.new_session(request)
    session['state'] = state
    session['next'] = next
    session['caller'] = 'signup'

    return aiohttp.web.HTTPFound(authorization_url)


async def _wait_websocket(request, email):
    app = request.app
    db = app['db']

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    user = await user_from_email(db, email)
    if not user:
        return web.HTTPNotFound()

    # forward authorization
    headers = {}
    if 'Authorization' in request.headers:
        headers['Authorization'] = request.headers['Authorization']
    if 'X-Hail-Internal-Authorization' in request.headers:
        headers['X-Hail-Internal-Authorization'] = request.headers['X-Hail-Internal-Authorization']

    cookies = {}
    if 'session' in request.cookies:
        cookies['session'] = request.cookies['session']
    if 'sesh' in request.cookies:
        cookies['sesh'] = request.cookies['sesh']

    count = 0
    while count < 10:
        try:
            updated_user = await user_from_email(db, email)
            changed = user['state'] != updated_user['state']
            if changed:
                log.info(f"user {user['username']} status changed: {user['state']} => {updated_user['state']}")
                break
        except Exception:  # pylint: disable=broad-except
            log.exception(f"/signup/wait: error while updating status for user {user['username']}")
        await asyncio.sleep(1)
        count += 1

    ready = updated_user and updated_user['state'] == 'active'

    # 0/1 ready
    await ws.send_str(str(int(ready)))

    return ws


@routes.get('/signup/wait')
@web_maybe_authenticated_user
async def account_pending(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    if 'pending' not in session:
        raise web.HTTPUnauthorized()
    return await _wait_websocket(request, session['email'])


@routes.get('/login')
@web_maybe_authenticated_user
async def login(request, userdata):
    db = request.app['db']

    next = request.query.get('next', deploy_config.external_url('notebook', ''))
    session = await aiohttp_session.get_session(request)

    if userdata:
        return aiohttp.web.HTTPFound(next)

    if 'pending' in session:
        email = session['email']
        user = await user_from_email(db, email)

        if user is None:
            cleanup_session(session)
            return aiohttp.web.HTTPFound(next)

        if user['state'] == 'deleting' or user['state'] == 'deleted':
            cleanup_session(session)
            set_message(session, f'Account for user {user["username"]} is deleted.', 'error')
            return aiohttp.web.HTTPFound(next)

        if user['state'] == 'active':
            session_id = await create_session(db, user['id'])
            cleanup_session(session)
            session['session_id'] = session_id
            return aiohttp.web.HTTPFound(next)

        assert user['state'] == 'creating'

        page_context = {
            'username': user['username'],
            'email': email,
            'state': user['state']
        }

        set_message(session, f'Account for user {user["username"]} is not active', 'error')
        return await render_template('auth', request, userdata, 'signup.html', page_context)

    flow = get_flow(deploy_config.external_url('auth', '/oauth2callback'))

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    session = await aiohttp_session.new_session(request)
    session['state'] = state
    session['next'] = next
    session['caller'] = 'login'

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
    except Exception as e:
        log.exception('oauth2 callback: could not fetch and verify token')
        raise web.HTTPUnauthorized() from e

    db = request.app['db']

    user = await user_from_email(db, email)

    caller = session['caller']

    if caller == 'login':
        if user is None:
            set_message(session, f'Account does not exist for email {email}', 'error')
        elif user['state'] != 'active':
            set_message(session, f'Account for email {email} is not active', 'error')
        else:
            session_id = await create_session(db, user['id'])
            session['session_id'] = session_id

        del session['state']
        next = session.pop('next')
        return aiohttp.web.HTTPFound(next)

    assert caller == 'signup'

    if user is None:
        # FIXME: come up with user naming scheme once we allow non-Broad users to self-signup
        username, domain = email.split('@')

        # FIXME: remove personal email address
        if domain == 'broadinstitute.org' or email == 'jackie.i.goldstein@gmail.com':
            await db.execute_insertone(
                '''
            INSERT INTO users (state, username, email, is_developer)
            VALUES (%s, %s, %s, %s);
            ''',
                ('creating', username, email, False))

            session['pending'] = True
            session['email'] = email

            return aiohttp.web.HTTPFound(deploy_config.external_url('auth', '/signup'))
        raise web.HTTPUnauthorized()

    set_message(session, f'Account already exists for {email}', 'error')

    del session['state']
    next = session.pop('next')
    return aiohttp.web.HTTPFound(next)


@routes.get('/user')
@web_authenticated_users_only()
async def user_page(request, userdata):
    return await render_template('auth', request, userdata, 'user.html', {})


async def create_copy_paste_token(db, session_id, max_age_secs=300):
    copy_paste_token = secret_alnum_string()
    await db.just_execute(
        "INSERT INTO copy_paste_tokens (id, session_id, max_age_secs) VALUES(%s, %s, %s);",
        (copy_paste_token, session_id, max_age_secs))
    return copy_paste_token


@routes.post('/copy-paste-token')
@check_csrf_token
@web_authenticated_users_only()
async def get_copy_paste_token(request, userdata):
    session = await aiohttp_session.get_session(request)
    session_id = session['session_id']
    db = request.app['db']
    copy_paste_token = await create_copy_paste_token(db, session_id)
    page_context = {
        'copy_paste_token': copy_paste_token
    }
    return await render_template('auth', request, userdata, 'copy-paste-token.html', page_context)


@routes.post('/api/v1alpha/copy-paste-token')
@rest_authenticated_users_only
async def get_copy_paste_token_api(request, userdata):
    session_id = userdata['session_id']
    db = request.app['db']
    copy_paste_token = await create_copy_paste_token(db, session_id)
    return web.Response(body=copy_paste_token)


@routes.post('/logout')
@check_csrf_token
@web_maybe_authenticated_user
async def logout(request, userdata):
    if not userdata:
        return web.HTTPFound(deploy_config.external_url('notebook', ''))

    db = request.app['db']
    session_id = userdata['session_id']
    await db.just_execute('DELETE FROM sessions WHERE session_id = %s;', session_id)

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
    db = request.app['db']
    users = [x async for x in
             db.select_and_fetchall('SELECT * FROM users;')]
    page_context = {
        'users': users
    }
    return await render_template('auth', request, userdata, 'users.html', page_context)


@routes.post('/users')
@check_csrf_token
@web_authenticated_developers_only()
async def post_create_user(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    db = request.app['db']
    post = await request.post()
    username = post['username']
    email = post['email']
    is_developer = post.get('is_developer') == '1'

    user_id = await db.execute_insertone(
        '''
INSERT INTO users (state, username, email, is_developer)
VALUES (%s, %s, %s, %s);
''',
        ('creating', username, email, is_developer))

    set_message(session, f'Created user {user_id} {username} {email}.', 'info')

    return web.HTTPFound(deploy_config.external_url('auth', '/users'))


@routes.post('/users/delete')
@check_csrf_token
@web_authenticated_developers_only()
async def delete_user(request, userdata):  # pylint: disable=unused-argument
    session = await aiohttp_session.get_session(request)
    db = request.app['db']
    post = await request.post()
    id = post['id']
    username = post['username']

    n_rows = await db.execute_update(
        '''
UPDATE users
SET state = 'deleting'
WHERE id = %s AND username = %s;
''',
        (id, username))
    if n_rows != 1:
        assert n_rows == 0
        set_message(session, f'Delete failed, no such user {id} {username}.', 'error')
    else:
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
    except Exception as e:
        log.exception('fetching and decoding token')
        raise web.HTTPUnauthorized() from e

    db = request.app['db']
    users = [x async for x in
             db.select_and_fetchall("SELECT * FROM users WHERE email = %s AND state = 'active';", email)]

    if len(users) != 1:
        raise web.HTTPUnauthorized()
    user = users[0]

    session_id = await create_session(db, user['id'], max_age_secs=None)

    return web.json_response({
        'token': session_id,
        'username': user['username']
    })


@routes.post('/api/v1alpha/copy-paste-login')
async def rest_copy_paste_login(request):
    copy_paste_token = request.query['copy_paste_token']
    db = request.app['db']

    @transaction(db)
    async def maybe_pop_token(tx):
        session = await tx.execute_and_fetchone("""
SELECT sessions.session_id AS session_id, users.username AS username FROM copy_paste_tokens
INNER JOIN sessions ON sessions.session_id = copy_paste_tokens.session_id
INNER JOIN users ON users.id = sessions.user_id
WHERE copy_paste_tokens.id = %s
  AND NOW() < TIMESTAMPADD(SECOND, copy_paste_tokens.max_age_secs, copy_paste_tokens.created)
  AND users.state = 'active';""", copy_paste_token)
        if session is None:
            raise web.HTTPUnauthorized()
        await tx.just_execute("DELETE FROM copy_paste_tokens WHERE id = %s;", copy_paste_token)
        return session

    session = await maybe_pop_token()  # pylint: disable=no-value-for-parameter
    return web.json_response({
        'token': session['session_id'],
        'username': session['username']
    })


@routes.post('/api/v1alpha/logout')
@rest_authenticated_users_only
async def rest_logout(request, userdata):
    session_id = userdata['session_id']
    db = request.app['db']
    await db.just_execute('DELETE FROM sessions WHERE session_id = %s;', session_id)

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

    db = request.app['db']
    users = [x async for x in
             db.select_and_fetchall('''
SELECT users.*, sessions.session_id FROM users
INNER JOIN sessions ON users.id = sessions.user_id
WHERE users.state = 'active' AND (sessions.session_id = %s) AND (ISNULL(sessions.max_age_secs) OR (NOW() < TIMESTAMPADD(SECOND, sessions.max_age_secs, sessions.created)));
''', session_id)]

    if len(users) != 1:
        log.info(f'Unknown session id: {session_id}')
        raise web.HTTPUnauthorized()
    user = users[0]

    return web.json_response(user)


async def on_startup(app):
    db = Database()
    await db.async_init(maxsize=50)
    app['db'] = db


async def on_cleanup(app):
    await app['db'].async_close()


def run():
    app = web.Application()

    setup_aiohttp_jinja2(app, 'auth')
    setup_aiohttp_session(app)

    setup_common_static_routes(routes)
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(deploy_config.prefix_application(app, 'auth'),
                host='0.0.0.0',
                port=5000,
                access_log_class=AccessLogger,
                ssl_context=get_in_cluster_server_ssl_context())
