import os
import uvloop
import asyncio
import aiodns
import aiohttp
from aiohttp import web
from kubernetes_asyncio import client, config
import logging

uvloop.install()

def make_logger():
    fmt = logging.Formatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
        '%(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    log = logging.getLogger('router-resolver')
    log.setLevel(logging.INFO)

    logging.basicConfig(handlers=[stream_handler], level=logging.INFO)

    return log

log = make_logger()

app = web.Application()
routes = web.RouteTableDef()

@routes.get('/')
async def auth(request):
    app = request.app
    k8s_client = app['k8s_client']
    host = request.headers['Host']
    if not host.endswith('.internal.hail.is'):
        # would prefer 404, but that's not handled by nginx request_auth
        return web.Response(status=403)
    labels = host.split('.')
    namespace = labels[-4]
    try:
        router = await k8s_client.read_namespaced_service('router', namespace)
    except client.rest.ApiException as err:
        if err.status == 404:
            return web.Response(status=403)
        raise
    return web.Response(status=200, headers={'X-Router-IP': router.spec.cluster_ip})

app.add_routes(routes)

async def on_startup(app):
    if 'BATCH_USE_KUBE_CONFIG' in os.environ:
        await config.load_kube_config()
    else:
        config.load_incluster_config()
    app['k8s_client'] = client.CoreV1Api()

app.on_startup.append(on_startup)

web.run_app(app, host='0.0.0.0', port=5000)
