import json
import asyncio
from aiohttp import web

done = asyncio.Event()
routes = web.RouteTableDef()


@routes.post('/status')
async def status(request):
    status = await request.json()
    print(f'status {json.dumps(status)}')
    return web.Response()


@routes.post('/terminate')
async def terminate(request):
    done.set()
    return web.Response()


@routes.post('/shutdown')
async def shutdown(request):
    b = await request.json()
    token = b['token']
    print(f'shutdown {token}')
    return web.Response()


async def main():
    app = web.Application()
    app.add_routes(routes)

    runner = web.AppRunner(app)
    app['runner'] = runner

    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 5000)
    await site.start()

    print('running...')
    await done.wait()

    await runner.cleanup()
    print('done.')


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
