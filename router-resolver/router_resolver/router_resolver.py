import os
import uvloop
from aiohttp import web
import aiohttp_session
from kubernetes_asyncio import client, config
import logging
from hailtop.auth import async_get_userinfo
from gear import configure_logging, setup_aiohttp_session

uvloop.install()

configure_logging()
log = logging.getLogger('router-resolver')

app = web.Application()
setup_aiohttp_session(app)

routes = web.RouteTableDef()


@routes.get('/auth/{namespace}')
async def auth(request):
    app = request.app
    k8s_client = app['k8s_client']
    namespace = request.match_info['namespace']

    headers = {}
    if 'X-Hail-Internal-Authorization' in request.headers:
        headers['Authorization'] = request.headers['X-Hail-Internal-Authorization']
    elif 'Authorization' in request.headers:
        headers['Authorization'] = request.headers['Authorization']
    else:
        session = await aiohttp_session.get_session(request)
        session_id = session.get('session_id')
        if not session_id:
            raise web.HTTPUnauthorized()
        headers['Authorization'] = f'Bearer {session_id}'

    userdata = await async_get_userinfo(headers=headers)
    if userdata['is_developer'] != 1:
        raise web.HTTPUnauthorized()

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
