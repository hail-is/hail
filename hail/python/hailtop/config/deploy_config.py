from typing import List, Tuple, Optional
import aiohttp
import random
import os
import json
import logging
from aiohttp import web
from ..utils import retry_transient_errors, first_extant_file
from ..tls import get_context_specific_client_ssl_context

log = logging.getLogger('deploy_config')


class DeployConfig:
    @staticmethod
    def from_config(config) -> 'DeployConfig':
        return DeployConfig(config['location'], config['default_namespace'], config['service_namespace'])

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
                'service_namespace': {}
            }
        return DeployConfig.from_config(config)

    def __init__(self, location: str, default_namespace: str, service_namespace: str):
        assert location in ('external', 'k8s', 'gce')
        self._location = location
        self._default_namespace = default_namespace
        self._service_namespace = service_namespace

    def with_service(self, service, ns) -> 'DeployConfig':
        return DeployConfig(self._location, self._default_namespace, {**self._service_namespace, service: ns})

    def location(self) -> str:
        return self._location

    def service_ns(self, service: str) -> str:
        return self._service_namespace.get(service, self._default_namespace)

    def scheme(self, base_scheme: str = 'http') -> str:
        # FIXME: should depend on ssl context
        return (base_scheme + 's') if self._location in ('external', 'k8s') else base_scheme

    def domain(self, service: str) -> str:
        ns = self.service_ns(service)
        if self._location == 'k8s':
            return f'{service}.{ns}'
        if self._location == 'gce':
            if ns == 'default':
                return f'{service}.hail'
            return 'internal.hail'
        assert self._location == 'external'
        if ns == 'default':
            return f'{service}.hail.is'
        return 'internal.hail.is'

    def base_path(self, service: str) -> str:
        ns = self.service_ns(service)
        if ns == 'default':
            return ''
        return f'/{ns}/{service}'

    def base_url(self, service: str, base_scheme: str = 'http') -> str:
        return f'{self.scheme(base_scheme)}://{self.domain(service)}{self.base_path(service)}'

    def url(self, service: str, path: str, base_scheme: str = 'http') -> str:
        return f'{self.base_url(service, base_scheme=base_scheme)}{path}'

    def auth_session_cookie_name(self) -> str:
        auth_ns = self.service_ns('auth')
        if auth_ns == 'default':
            return 'session'
        return 'sesh'

    def external_url(self, service: str, path: str, base_scheme: str = 'http') -> str:
        ns = self.service_ns(service)
        if ns == 'default':
            return f'{base_scheme}s://{service}.hail.is{path}'
        return f'{base_scheme}s://internal.hail.is/{ns}/{service}{path}'

    def prefix_application(self, app, service: str, **kwargs):
        base_path = self.base_path(service)
        if not base_path:
            return app

        root_routes = web.RouteTableDef()

        @root_routes.get('/healthcheck')
        async def get_healthcheck(request):  # pylint: disable=unused-argument,unused-variable
            return web.Response()

        root_app = web.Application(**kwargs)
        root_app.add_routes(root_routes)
        root_app.add_subapp(base_path, app)

        return root_app

    ADDRESS_HOSTS = ['shuffler', 'address']

    async def addresses(self, service: str) -> List[Tuple[str, int]]:
        if service not in DeployConfig.ADDRESS_HOSTS:
            return []

        from ..auth import service_auth_headers  # pylint: disable=cyclic-import,import-outside-toplevel
        namespace = self.service_ns(service)
        headers = service_auth_headers(self, namespace)
        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    ssl=get_context_specific_client_ssl_context()),
                raise_for_status=True,
                timeout=aiohttp.ClientTimeout(total=5),
                headers=headers) as session:
            async with await retry_transient_errors(
                    session.get,
                    self.url('address', f'/api/{service}')) as resp:
                dicts = await resp.json()
                return [(d['address'], d['port']) for d in dicts]

    async def maybe_address(self, service: str) -> Optional[Tuple[str, int]]:
        service_addresses = await self.addresses(service)
        n = len(service_addresses)
        if n == 0:
            return None
        return service_addresses[random.randrange(0, n)]

    async def address(self, service: str) -> Tuple[str, int]:
        address = await self.maybe_address(service)
        assert address is not None
        return address


deploy_config: Optional[DeployConfig] = None


def get_deploy_config() -> DeployConfig:
    global deploy_config

    if not deploy_config:
        deploy_config = DeployConfig.from_config_file()
    return deploy_config
