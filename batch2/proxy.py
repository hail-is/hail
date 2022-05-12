import asyncio

from aiohttp import web

from hailtop.batch_client.aioclient import BatchClient

routes = web.RouteTableDef()


@routes.get('/api/{route:.*}')
async def proxy_api(request):
    client: BatchClient = app['batch_client']
    route = request.match_info['route']
    data = await client._get(f'/api/{route}')
    return web.json_response(await data.json())


async def on_startup(app):
    app['batch_client'] = await BatchClient.create('test')


app = web.Application()
app.add_routes(routes)
app.on_startup.append(on_startup)
web.run_app(app, host='0.0.0.0', port=5050)
