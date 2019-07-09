import re
import logging
import asyncio
import datetime
import aiohttp
from aiohttp import web
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

app = web.Application()
routes = web.RouteTableDef()

CLIENT_SECRET = '/auth-oauth2-client-secret/client_secret.json'
SCOPES = [
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
    'openid'
]


def get_flow(state=None):
  return google_auth_oauthlib.flow.Flow.from_client_secrets_file(
    CLIENT_SECRET, scopes=SCOPES, state=state)


@routes.get('/')
async def get_index(request):
    return aiohttp.web.HTTPFound('/login')


@routes.get('/login')
async def login(request):
    flow = get_flow()

    flow.redirect_uri = 'https://auth.hail.is/oauth2callback'

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    resp = aiohttp.web.HTTPFound(authorization_url)
    resp.set_cookie('state', state,
                    # 5m
                    max_age=300,
                    secure=True,
                    httponly=True)
    # FIXME put next, state in the session: https://github.com/aio-libs/aiohttp-session
    next = request.query.get('next')
    if next:
        resp.set_cookie('next', next,
                        max_age=300,
                        secure=True,
                        httponly=True)
    return resp


@routes.post('/logout')
async def logout():
    return {}


@routes.get('/oauth2callback')
async def callback(request):
    authorization_response = str(request.url)
    authorization_response = re.sub('^http://', 'https://', authorization_response)

    state = request.cookies['state']

    flow = get_flow(state=state)
    flow.redirect_uri = 'https://auth.hail.is/oauth2callback'
    flow.fetch_token(authorization_response=authorization_response)

    decoded = google.oauth2.id_token.verify_oauth2_token(
        flow.credentials.id_token, google.auth.transport.requests.Request())

    # FIXME
    log.info(f'decoded: {decoded}')

    session_id = secrets.token_bytes(32)

    next = request.cookies.get('next', 'https://notebook2.hail.is')
    resp = aiohttp.web.HTTPFound(next)
    # clear state
    resp.set_cookie('state', '', max_age=0, secure=True, httponly=True)
    resp.set_cookie('next', '', max_age=0, secure=True, httponly=True)
    resp.set_cookie('session', base64.b64encode(session_id).decode('ascii'),
                    # 30d
                    max_age=2592000,
                    secure=True, httponly=True)
    return resp


app.add_routes(routes)


def run():
    configure_logging()
    web.run_app(app, host='0.0.0.0', port=5000)
