from typing import List, Tuple, Optional, Dict
import aiohttp
import random
import os
import json
import logging
from aiohttp import web
from ..utils import retry_transient_errors, first_extant_file
from ..tls import internal_client_ssl_context

log = logging.getLogger('deploy_config')


class DeployConfig:
    @staticmethod
    def from_config(config: dict) -> 'DeployConfig':
        domain = config.get('domain', 'hail.is')
        return DeployConfig(config['location'], config['default_namespace'], domain)

    def get_config(self):
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
                'domain': 'hail.is'
            }
        return DeployConfig.from_config(config)

    def __init__(self, location: str, default_namespace: str, domain: str):
        assert location in ('external', 'k8s', 'gce')
        self._location = location
        self._default_namespace = default_namespace
        self._domain = domain

    def with_default_namespace(self, default_namespace):
        return DeployConfig(self._location, default_namespace, self._domain)

    def default_namespace(self):
        return self._default_namespace

    def location(self) -> str:
        return self._location

    def service_ns(self, service):  # pylint: disable=unused-argument
        return self._default_namespace

    def scheme(self, service: str, base_scheme: str = 'http', use_address: bool = False) -> str:
        if use_address and service in DeployConfig.ADDRESS_SERVICES:
            return base_scheme + 's'
        if self._location != 'gce':
            return base_scheme + 's'
        return base_scheme

    def domain(self, service: str, use_address: bool = False) -> str:
        ns = self.service_ns(service)
        if self._location == 'k8s':
            return f'{service}.{ns}'
        if self._location == 'gce':
            if use_address and service in DeployConfig.ADDRESS_SERVICES:
                return f'{service}.{ns}'
            if ns == 'default':
                return f'{service}.hail'
            return 'internal.hail'
        assert self._location == 'external'
        if ns == 'default':
            return f'{service}.{self._domain}'
        return f'internal.{self._domain}'

    def base_path(self, service: str) -> str:
        ns = self.service_ns(service)
        if ns == 'default':
            return ''
        return f'/{ns}/{service}'

    def base_url(self, service: str, *, base_scheme: str = 'http', use_address: bool = False) -> str:
        scheme = self.scheme(service, base_scheme=base_scheme, use_address=use_address)
        domain = self.domain(service, use_address=use_address)
        return f'{scheme}://{domain}{self.base_path(service)}'

    def url(self, service: str, path: str, *, base_scheme: str = 'http', use_address: bool = False) -> str:
        return f'{self.base_url(service, base_scheme=base_scheme, use_address=use_address)}{path}'

    def auth_session_cookie_name(self) -> str:
        auth_ns = self.service_ns('auth')
        if auth_ns == 'default':
            return 'session'
        return 'sesh'

    def external_url(self, service: str, path: str, base_scheme: str = 'http') -> str:
        ns = self.service_ns(service)
        if ns == 'default':
            if service == 'www':
                return f'{base_scheme}s://{self._domain}{path}'
            return f'{base_scheme}s://{service}.{self._domain}{path}'
        return f'{base_scheme}s://internal.{self._domain}/{ns}/{service}{path}'

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

    ADDRESS_SERVICES = ['shuffler', 'address']

    async def addresses(self, domain: str) -> List[Tuple[str, int]]:
        assert self._location != 'internal'

        domain_parts = domain.split('.')
        n_parts = len(domain_parts)
        assert n_parts > 0
        service = domain_parts[0]

        if n_parts > 2 or service not in DeployConfig.ADDRESS_SERVICES:
            return []
        if n_parts == 2 and domain_parts[1] == 'hail':
            # internal.hail, etc.
            return []

        if n_parts == 1:
            namespace = self.service_ns(service)
        elif n_parts == 2:
            namespace = domain_parts[1]

        from ..auth import service_auth_headers  # pylint: disable=cyclic-import,import-outside-toplevel
        headers = service_auth_headers(self, namespace)
        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    ssl=internal_client_ssl_context()),
                raise_for_status=True,
                timeout=aiohttp.ClientTimeout(total=5),
                headers=headers) as session:
            async with await retry_transient_errors(
                    session.get,
                    self.url('address', f'/api/{service}', use_address=False)) as resp:
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
