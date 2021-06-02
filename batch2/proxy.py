import asyncio
from aiohttp import web
from hailtop.batch_client.aioclient import BatchClient

routes = web.RouteTableDef()

client = BatchClient('test')


@routes.get('/api/{route:.*}')
async def proxy_api(request):
    route = request.match_info['route']
    data = await client._get(f'/api/{route}')
    return web.json_response(await data.json())


app = web.Application()
app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=5050)
