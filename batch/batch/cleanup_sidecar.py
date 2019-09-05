import logging
import os
import subprocess as sp

from aiohttp import web
from hailtop import gear

from .stoppable_server import StoppableServer

copy_output_cmd = os.environ.get('COPY_OUTPUT_CMD')

gear.configure_logging()
log = logging.getLogger('cleanup_sidecar')

app = web.Application()
routes = web.RouteTableDef()
server = StoppableServer(app, '0.0.0.0', 5000)


@routes.post('/')
async def finish(request):
    del request
    if copy_output_cmd is not None:
        log.info(f'copying out data')
        try:
            copy_output = sp.check_output(copy_output_cmd, shell=True, stderr=sp.STDOUT)
            log.info(copy_output.decode('ascii'))
        except sp.CalledProcessError as err:
            log.error(f'bad exit code {err.returncode}: {err.output}')
            log.exception(f'exiting 1 due to exception')
            server.stop(1)
    log.info(f'exiting cleanly')
    await server.stop(0)
    return web.Response()

if __name__ == '__main__':
    app.add_routes(routes)
    server.run()
