import os
import uvloop
import asyncio
import aiodns
import aiohttp
from aiohttp import web
from kubernetes_asyncio import client, config
import logging

from hailtop.gear import configure_logging, get_deploy_config

uvloop.install()

configure_logging()
log = logging.getLogger('router-resolver')

app = web.Application()

routes = web.RouteTableDef()


@routes.get('/auth/{namespace}')
async def auth(request):
    deploy_config = get_deploy_config()
    app = request.app
    k8s_client = app['k8s_client']
    namespace = request.match_info['namespace']

    headers = {}
    cookies = {}
    if 'X-Hail-Internal-Authorization' in request.headers:
        headers['Authorization'] = request.headers['X-Hail-Internal-Authorization']
    elif 'Authorization' in request.headers:
        headers['Authorization'] = request.headers['Authorization']
    elif 'session' in request.cookies:
        cookies['session'] = request.cookies['session']
    else:
        raise web.HTTPUnauthorized()

    try:
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(deploy_config.url('auth', '/api/v1alpha/userinfo'),
                                   headers=headers, cookies=cookies) as resp:
                userdata = await resp.json()
    except Exception:
        log.exception('getting userinfo')
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
