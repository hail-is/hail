from typing import List, DefaultDict, Generic, TypeVar, Dict, Tuple, Any, NamedTuple
import time
import secrets
import logging
import json
import copy
from functools import wraps
import concurrent
import asyncio
from aiohttp import web
import aiohttp_session
import kubernetes_asyncio as kube
import google.oauth2.service_account
from gear import Database, setup_aiohttp_session, web_authenticated_developers_only, \
    check_csrf_token, transaction, AccessLogger
from hailtop.config import get_deploy_config
from hailtop.utils import time_msecs, RateLimit
from hailtop.tls import get_server_ssl_context
from hailtop import aiogoogle
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template, \
    set_message
import uvloop
import sortedcontainers
from collections import defaultdict


uvloop.install()

log = logging.getLogger('address')

routes = web.RouteTableDef()

deploy_config = get_deploy_config()


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('/')
@routes.get('')
@web_authenticated_developers_only()
async def get_index(request, userdata):
    cache = request.app['cache']
    keys = sorted(list(cache.keys))
    page_context = {
        'attributes': [
            ['VALID_DURATION_IN_SECONDS', VALID_DURATION_IN_SECONDS]],
        'rows': [{'namespace': k[0],
                  'name': k[1],
                  'addresses': ", ".join([x.ip for x in cache.entries[k].value]),
                  'ports': ", ".join([x.port for x in cache.entries[k].value]),
                  'lifetime': time.time() - cache.entries[k].expire_time,
                  'lock': k in cache.locks}
                 for k in keys]
    }
    return await render_template('address', request, userdata, 'index.html', page_context)


@routes.get('/api/{namespace}/{name}')
@web_authenticated_developers_only()
async def get_name(request, userdata):
    namespace = request.match_info['namespace']
    name = request.match_info['name']
    log.info(f'get {namespace} {name}')
    result = await request.app['cache'].get(name, namespace)
    log.info(f'result {result}')
    return web.json_response(result)


VALID_DURATION_IN_SECONDS = 60


T = TypeVar('T')


class CacheEntry(Generic[T]):
    def __init__(self, value: T):
        self.value: T = value
        self.expire_time: float = time.time() + VALID_DURATION_IN_SECONDS


class AddressAndPort(NamedTuple):
    address: str
    port: int


class Cache():
    def __init__(self, k8s_client: kube.client.CoreV1Api):
        self.entries: Dict[Tuple[str, str], CacheEntry[List[AddressAndPort]]] = dict()
        self.locks: DefaultDict[Tuple[str, str], asyncio.Lock] = defaultdict(asyncio.Lock)
        self.keys = sortedcontainers.SortedSet(
            key=lambda key: self.entries[key].expire_time)
        self.k8s_client: kube.client.CoreV1Api = k8s_client

    async def get(self, name: str, namespace: str):
        key = (namespace, name)
        if key in self.entries:
            entry = self.entries[key]
            if entry.expire_time >= time.time():
                log.info(f'hit cache for {key}')
                return entry.value
            log.info(f'stale cache for {key}')
        lock = self.locks[key]
        log.info(f'lock {lock}')
        async with lock:
            if key in self.entries:
                entry = self.entries[key]
                if entry.expire_time >= time.time():
                    return entry.value

            asyncio.ensure_future(self.maybe_remove_one_old_entry())
            k8s_endpoints = await self.k8s_client.read_namespaced_endpoints(
                name, namespace)
            endpoints = [AddressAndPort(ip.ip, port.port)
                         for endpoint in k8s_endpoints.subsets
                         for port in endpoint.ports
                         for ip in endpoint.addresses]
            self.entries[key] = CacheEntry(endpoints)
            self.keys.add(key)
            log.info(f'fetched new value for {key}: {endpoints}')
            return endpoints

    async def maybe_remove_one_old_entry(self):
        if len(self.keys) > 0:
            key = self.keys.pop()
            if self.entries[key].expire_time < time.time():
                del self.entries[key]
                del self.locks[key]
            else:
                self.keys.add(key)


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool

    kube.config.load_incluster_config()
    k8s_client = kube.client.CoreV1Api()
    app['cache'] = Cache(k8s_client)


async def on_cleanup(app):
    blocking_pool = app['blocking_pool']
    blocking_pool.shutdown()


def run():
    app = web.Application()
    setup_aiohttp_session(app)

    setup_aiohttp_jinja2(app, 'address')
    setup_common_static_routes(routes)
    app.add_routes(routes)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(deploy_config.prefix_application(app, 'address'),
                host='0.0.0.0',
                port=5000,
                access_log_class=AccessLogger,
                ssl_context=get_server_ssl_context())
