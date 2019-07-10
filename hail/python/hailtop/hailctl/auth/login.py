import os
import socket
import asyncio
import aiohttp
from aiohttp import web


def init_parser(parser):
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


async def handle_callback(runner):
    code = await q.get()
    return code


async def auth_flow(session):
    runner, port = await start_server()

    async with session.get('https://auth.hail.is/api/v1alpha/login',
                           params={'callback_port': port}) as resp:
        resp = await resp.json()
    authorization_url = resp['authorization_url']
    state = resp['state']

    # FIXME use webbrowser module
    print(f'''
To log into Hail using Google, in your browser, visit:

    {authorization_url}
''')

    code = await runner.app['q'].get()
    await runner.cleanup()

    async with session.get('https://auth.hail.is/api/v1alpha/oauth2callback',
                           params={
                               'callback_port': port,
                               'code': code,
                               'state': state
                           }) as resp:
        resp = await resp.json()
    token = resp['token']
    username = resp['username']

    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)

    token_file = (os.environ.get('HAIL_TOKEN_FILE')
                  or os.path.expanduser('~/.hail/token'))

    with open(token_file, 'w') as f:
        f.write(token)
    os.chmod(token_file, 0o600)

    print(f'Logged in as {username}.')


async def async_main():
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        await auth_flow(session)

def main(args, pass_through_args):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
