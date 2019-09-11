import logging

from aiohttp import web
from gear import configure_logging

from .stoppable_server import StoppableServer

configure_logging()
log = logging.getLogger('keep_alive_sidecar')

app = web.Application()
routes = web.RouteTableDef()
server = StoppableServer(app, '0.0.0.0', 5001)


@routes.post('/')
async def finish(request):
    del request
    log.info(f'received shutdown request')
    await server.stop(0)
    return web.Response()

if __name__ == '__main__':
    app.add_routes(routes)
    server.run()
