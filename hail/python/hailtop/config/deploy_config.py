from typing import Dict, Optional, TypeVar, Union
import os
import ssl
import json
import logging
from ..utils import first_extant_file
from ..tls import external_client_ssl_context, internal_client_ssl_context, internal_server_ssl_context

from .user_config import get_user_config

log = logging.getLogger('deploy_config')

T = TypeVar("T")


def env_var_or_default(name: str, default: T) -> Union[str, T]:
    return os.environ.get(f'HAIL_{name.upper()}', default)


class DeployConfig:
    @classmethod
    def from_config(cls, config: Dict[str, str]) -> 'DeployConfig':
        location = env_var_or_default('location', config['location'])
        domain = env_var_or_default('domain', config['domain'])
        ns = env_var_or_default('default_namespace', config['default_namespace'])
        base_path = env_var_or_default('base_path', config.get('base_path')) or None
        if base_path is None and ns != 'default':
            domain = f'internal.{config["domain"]}'
            base_path = f'/{ns}'

        return cls(location, ns, domain, base_path)

    def get_config(self) -> Dict[str, Optional[str]]:
        return {
            'location': self._location,
            'default_namespace': self._default_namespace,
            'domain': self._domain,
            'base_path': self._base_path,
        }

    @classmethod
    def from_config_file(cls, config_file=None) -> 'DeployConfig':
        config_file = first_extant_file(
            config_file,
            os.environ.get('HAIL_DEPLOY_CONFIG_FILE'),
            os.path.expanduser('~/.hail/deploy-config.json'),
            '/deploy-config/deploy-config.json',
        )
        if config_file is not None:
            log.info(f'deploy config file found at {config_file}')
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            log.info(f'deploy config location: {config["location"]}')
        else:
            log.info(f'deploy config file not found: {config_file}')
            config = {
                'location': 'external',
                'default_namespace': 'default',
                'domain': get_user_config().get('global', 'domain', fallback='hail.is'),
            }
        return cls.from_config(config)

    def __init__(self, location: str, default_namespace: str, domain: str, base_path: Optional[str]):
        assert location in ('external', 'k8s', 'gce')
        self._location = location
        self._default_namespace = default_namespace
        self._domain = domain
        self._base_path = base_path

    def with_default_namespace(self, default_namespace):
        return DeployConfig(self._location, default_namespace, self._domain, self._base_path)

    def with_location(self, location):
        return DeployConfig(location, self._default_namespace, self._domain, self._base_path)

    def default_namespace(self):
        return self._default_namespace

    def location(self):
        return self._location

    def scheme(self, base_scheme='http'):
        # FIXME: should depend on ssl context
        return (base_scheme + 's') if self._location in ('external', 'k8s') else base_scheme

    def domain(self, service):
        ns = self._default_namespace
        if self._location == 'k8s':
            return f'{service}.{ns}'
        if self._location == 'gce':
            if self._base_path is None:
                return f'{service}.hail'
            return 'internal.hail'
        assert self._location == 'external'
        if self._base_path is None:
            return f'{service}.{self._domain}'
        return self._domain

    def base_path(self, service):
        if self._base_path is None:
            return ''
        return f'{self._base_path}/{service}'

    def base_url(self, service, base_scheme='http'):
        return f'{self.scheme(base_scheme)}://{self.domain(service)}{self.base_path(service)}'

    def url(self, service, path, base_scheme='http'):
        return f'{self.base_url(service, base_scheme=base_scheme)}{path}'

    def auth_session_cookie_name(self):
        if self._default_namespace == 'default':
            return 'session'
        return 'sesh'

    def external_url(self, service, path, base_scheme='http'):
        if self._base_path is None:
            if service == 'www':
                return f'{base_scheme}s://{self._domain}{path}'
            return f'{base_scheme}s://{service}.{self._domain}{path}'
        return f'{base_scheme}s://{self._domain}{self._base_path}/{service}{path}'

    def prefix_application(self, app, service, **kwargs):
        from aiohttp import web  # pylint: disable=import-outside-toplevel

        base_path = self.base_path(service)
        if not base_path:
            return app

        root_routes = web.RouteTableDef()

        @root_routes.get('/healthcheck')
        async def get_healthcheck(_):
            return web.Response()

        @root_routes.get('/metrics')
        async def get_metrics(_):
            raise web.HTTPFound(location=f'{base_path}/metrics')

        root_app = web.Application(**kwargs)
        root_app.add_routes(root_routes)
        root_app.add_subapp(base_path, app)

        log.info(f'serving paths at {base_path}')
        return root_app

    def client_ssl_context(self) -> ssl.SSLContext:
        if self._location == 'k8s':
            return internal_client_ssl_context()
        # no encryption on the internal gateway
        return external_client_ssl_context()

    def server_ssl_context(self) -> Optional[ssl.SSLContext]:
        if self._location == 'k8s':
            return internal_server_ssl_context()
        # local mode does not have access to self-signed certs
        return None


class TerraDeployConfig(DeployConfig):
    def domain(self, service):
        if self._location == 'k8s':
            return {
                'batch-driver': 'localhost:5000',
                'batch': 'localhost:5001',
            }[service]
        return self._domain

    def client_ssl_context(self) -> ssl.SSLContext:
        # Terra app networking doesn't use self-signed certs
        return external_client_ssl_context()

    def server_ssl_context(self) -> Optional[ssl.SSLContext]:
        # Terra app services are in the same pod and just use http
        return None


deploy_config = None


def get_deploy_config() -> DeployConfig:
    global deploy_config

    if not deploy_config:
        if os.environ['HAIL_TERRA']:
            deploy_config = TerraDeployConfig.from_config_file()
        else:
            deploy_config = DeployConfig.from_config_file()
    return deploy_config
