import uvloop
import asyncio
import aiodns
from aiohttp import web

uvloop.install()

app = web.Application()
routes = web.RouteTableDef()

@routes.get('/healthcheck')
async def healthcheck(request):
    return web.Response(status=200)

app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=5000)
