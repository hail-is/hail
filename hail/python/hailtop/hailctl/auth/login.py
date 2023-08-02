import os
import socket
import asyncio
import json
import webbrowser
from aiohttp import web

from typing import Optional


from hailtop.config import get_deploy_config
from hailtop.auth import get_tokens, hail_credentials
from hailtop.httpx import client_session


routes = web.RouteTableDef()


@routes.get('/oauth2callback')
async def callback(request):
    q = request.app['q']
    code = request.query['code']
    await q.put(code)
    # FIXME redirect a nice page like auth.hail.is/hailctl/authenciated with link to more information
    return web.Response(text='hailctl is now authenticated.')


async def start_server():
    app = web.Application()
    app['q'] = asyncio.Queue()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    _, port = sock.getsockname()
    site = web.SockSite(runner, sock, shutdown_timeout=0)
    await site.start()

    return (runner, port)


async def auth_flow(deploy_config, default_ns, session):
    runner, port = await start_server()

    async with session.get(deploy_config.url('auth', '/api/v1alpha/login'), params={'callback_port': port}) as resp:
        json_resp = await resp.json()

    flow = json_resp['flow']
    state = json_resp['state']
    authorization_url = flow['authorization_url']

    print(
        f'''
Visit the following URL to log into Hail:

    {authorization_url}

Opening in your browser.
'''
    )
    webbrowser.open(authorization_url)

    code = await runner.app['q'].get()
    await runner.cleanup()

    async with session.get(
        deploy_config.url('auth', '/api/v1alpha/oauth2callback'),
        params={
            'callback_port': port,
            'code': code,
            'state': state,
            'flow': json.dumps(flow),
        },
    ) as resp:
        json_resp = await resp.json()
    token = json_resp['token']
    username = json_resp['username']

    tokens = get_tokens()
    tokens[default_ns] = token
    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)
    tokens.write()

    if default_ns == 'default':
        print(f'Logged in as {username}.')
    else:
        print(f'Logged into namespace {default_ns} as {username}.')


async def async_login(namespace: Optional[str]):
    deploy_config = get_deploy_config()
    if namespace:
        deploy_config = deploy_config.with_default_namespace(namespace)
    namespace = namespace or deploy_config.default_namespace()
    headers = await hail_credentials(namespace=namespace, authorize_target=False).auth_headers()
    async with client_session(headers=headers) as session:
        await auth_flow(deploy_config, namespace, session)
