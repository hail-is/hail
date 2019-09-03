import os
import socket
import asyncio
import webbrowser
import aiohttp
from aiohttp import web

from hailtop.gear import get_deploy_config
from hailtop.gear.auth import get_tokens, auth_headers


def init_parser(parser):  # pylint: disable=unused-argument
    pass


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


async def auth_flow(session):
    deploy_config = get_deploy_config()

    runner, port = await start_server()

    async with session.get(deploy_config.url('auth', '/api/v1alpha/login'),
                           params={'callback_port': port}) as resp:
        resp = await resp.json()
    authorization_url = resp['authorization_url']
    state = resp['state']

    print(f'''
Visit the following URL to log into Hail with Google:

    {authorization_url}

Opening in your browser.
''')
    webbrowser.open(authorization_url)

    code = await runner.app['q'].get()
    await runner.cleanup()

    async with session.get(
            deploy_config.url('auth', '/api/v1alpha/oauth2callback'),
            params={
                'callback_port': port,
                'code': code,
                'state': state
            }) as resp:
        resp = await resp.json()
    token = resp['token']
    username = resp['username']

    auth_ns = deploy_config.service_ns('auth')
    tokens = get_tokens()
    tokens[auth_ns] = token
    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)
    tokens.write()

    print(f'Logged in as {username}.')


async def async_main():
    headers = auth_headers('auth', authorize_target=False)
    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60), headers=headers) as session:
        await auth_flow(session)


def main(args, pass_through_args):  # pylint: disable=unused-argument
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
