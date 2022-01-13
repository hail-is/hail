from typing import List, Tuple, Dict
import aiohttp
import random
import os
import json
import logging
from aiohttp import web
from ..utils import retry_transient_errors, first_extant_file
from ..tls import internal_client_ssl_context

from .user_config import get_user_config

log = logging.getLogger('deploy_config')


class DeployConfig:
    @staticmethod
    def from_config(config) -> 'DeployConfig':
        return DeployConfig(config['location'], config['default_namespace'], config['domain'])

    def get_config(self) -> Dict[str, str]:
        return {
            'location': self._location,
            'default_namespace': self._default_namespace,
            'domain': self._domain
        }

    @staticmethod
    def from_config_file(config_file=None) -> 'DeployConfig':
        config_file = first_extant_file(
            config_file,
            os.environ.get('HAIL_DEPLOY_CONFIG_FILE'),
            os.path.expanduser('~/.hail/deploy-config.json'),
            '/deploy-config/deploy-config.json')
        if config_file is not None:
            log.info(f'deploy config file found at {config_file}')
            with open(config_file, 'r') as f:
                config = json.load(f)
            log.info(f'deploy config location: {config["location"]}')
        else:
            log.info(f'deploy config file not found: {config_file}')
            config = {
                'location': 'external',
                'default_namespace': 'default',
                'domain': get_user_config().get('global', 'domain', fallback='hail.is'),
            }
        return DeployConfig.from_config(config)

    def __init__(self, location, default_namespace, domain):
        assert location in ('external', 'k8s', 'gce')
        self._location = location
        self._default_namespace = default_namespace
        self._domain = domain

    def with_default_namespace(self, default_namespace):
        return DeployConfig(self._location, default_namespace, self._domain)

    def default_namespace(self):
        return self._default_namespace

    def location(self):
        return self._location

    def service_ns(self, service):  # pylint: disable=unused-argument
        return self._default_namespace

    def scheme(self, base_scheme='http'):
        # FIXME: should depend on ssl context
        return (base_scheme + 's') if self._location in ('external', 'k8s') else base_scheme

    def domain(self, service):
        ns = self.service_ns(service)
        if self._location == 'k8s':
            return f'{service}.{ns}'
        if self._location == 'gce':
            if ns == 'default':
                return f'{service}.hail'
            return 'internal.hail'
        assert self._location == 'external'
        if ns == 'default':
            return f'{service}.{self._domain}'
        return f'internal.{self._domain}'

    def base_path(self, service):
        ns = self.service_ns(service)
        if ns == 'default':
            return ''
        return f'/{ns}/{service}'

    def base_url(self, service, base_scheme='http'):
        return f'{self.scheme(base_scheme)}://{self.domain(service)}{self.base_path(service)}'

    def url(self, service, path, base_scheme='http'):
        return f'{self.base_url(service, base_scheme=base_scheme)}{path}'

    def auth_session_cookie_name(self):
        auth_ns = self.service_ns('auth')
        if auth_ns == 'default':
            return 'session'
        return 'sesh'

    def external_url(self, service, path, base_scheme='http'):
        ns = self.service_ns(service)
        if ns == 'default':
            if service == 'www':
                return f'{base_scheme}s://{self._domain}{path}'
            return f'{base_scheme}s://{service}.{self._domain}{path}'
        return f'{base_scheme}s://internal.{self._domain}/{ns}/{service}{path}'

    def prefix_application(self, app, service, **kwargs):
        base_path = self.base_path(service)
        if not base_path:
            return app

        root_routes = web.RouteTableDef()

        @root_routes.get('/healthcheck')
        async def get_healthcheck(request):  # pylint: disable=unused-argument,unused-variable
            return web.Response()

        @root_routes.get('/metrics')
        async def get_metrics(request):  # pylint: disable=unused-argument,unused-variable
            return web.HTTPFound(location=f'{base_path}/metrics')

        root_app = web.Application(**kwargs)
        root_app.add_routes(root_routes)
        root_app.add_subapp(base_path, app)

        log.info(f'serving paths at {base_path}')
        return root_app

    async def addresses(self, service: str) -> List[Tuple[str, int]]:
        from ..auth import service_auth_headers  # pylint: disable=cyclic-import,import-outside-toplevel
        namespace = self.service_ns(service)
        headers = service_auth_headers(self, namespace)
        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=internal_client_ssl_context()),
                raise_for_status=True,
                timeout=aiohttp.ClientTimeout(total=5),
                headers=headers) as session:
            async with await retry_transient_errors(
                    session.get,
                    self.url('address', f'/api/{service}')) as resp:
                dicts = await resp.json()
                return [(d['address'], d['port']) for d in dicts]

    async def address(self, service: str) -> Tuple[str, int]:
        service_addresses = await self.addresses(service)
        n = len(service_addresses)
        assert n > 0
        return service_addresses[random.randrange(0, n)]


deploy_config = None


def get_deploy_config() -> DeployConfig:
    global deploy_config

    if not deploy_config:
        deploy_config = DeployConfig.from_config_file()
    return deploy_config
