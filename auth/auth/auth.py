import re
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


@routes.post('/logout')
async def logout():
    session = await aiohttp_session.get_session(request)
    if 'session_id' in session:
        del session['session_id']
    return aiohttp.web.HTTPFound('https://auth.hail.is/login')


@routes.get('/oauth2callback')
async def callback(request):
    session = await aiohttp_session.get_session(request)

    # FIXME check
    state = session['state']

    flow = get_flow(state=state)
    flow.fetch_token(authorization_response='https://auth.hail.is/oauth2callback')

    # FIXME check
    decoded = google.oauth2.id_token.verify_oauth2_token(
        flow.credentials.id_token, google.auth.transport.requests.Request())

    # FIXME
    log.info(f'decoded: {decoded}')

    session_id = secrets.token_bytes(32)

    del session['state']
    session['session_id'] = session_id
    next = session.get('next', 'https://notebook2.hail.is')
    if 'next' in session:
        del session['next']
    return aiohttp.web.HTTPFound(next)


@routes.post('/log')
async def logout():
    session = await aiohttp_session.get_session(request)
    log.info(f'session: {session}')
    return web.Response(status=200)


def run():
    configure_logging()
    app = web.Application()
    fernet_key = fernet.Fernet.generate_key()
    secret_key = base64.urlsafe_b64decode(fernet_key)
    aiohttp_session.setup(app, aiohttp_session.cookie_storage.EncryptedCookieStorage(secret_key, max_age=30 * 24 * 60 * 60))
    app.add_routes(routes)
    web.run_app(app, host='0.0.0.0', port=5000)
