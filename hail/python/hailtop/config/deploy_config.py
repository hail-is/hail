import os
import json
import logging
from aiohttp import web
from hailtop.utils import first_extant_file

log = logging.getLogger('gear')


class DeployConfig:
    @staticmethod
    def from_config(config):
        return DeployConfig(config['location'], config['default_namespace'], config['service_namespace'])

    @staticmethod
    def from_config_file(config_file=None):
        config_file = first_extant_file(
            config_file,
            os.environ.get('HAIL_DEPLOY_CONFIG_FILE'),
            os.path.expanduser('~/.hail/deploy-config.json'),
            '/deploy-config/deploy-config.json')
        if config_file is not None:
            with open(config_file, 'r') as f:
                config = json.loads(f.read())
        else:
            log.info(f'deploy config file not found: {config_file}')
            config = {
                'location': 'external',
                'default_namespace': 'default',
                'service_namespace': {}
            }
        return DeployConfig.from_config(config)

    def __init__(self, location, default_namespace, service_namespace):
        assert location in ('external', 'k8s', 'gce')
        self._location = location
        self._default_namespace = default_namespace
        self._service_namespace = service_namespace

    def with_service(self, service, ns):
        return DeployConfig(self._location, self._default_namespace, {**self._service_namespace, service: ns})

    def location(self):
        return self._location

    def service_ns(self, service):
        return self._service_namespace.get(service, self._default_namespace)

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
            return f'{service}.hail.is'
        return 'internal.hail.is'

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
            return f'{base_scheme}s://{service}.hail.is{path}'
        return f'{base_scheme}s://internal.hail.is/{ns}/{service}{path}'

    def prefix_application(self, app, service, **kwargs):
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


deploy_config = None


def get_deploy_config():
    global deploy_config

    if not deploy_config:
        deploy_config = DeployConfig.from_config_file()
    return deploy_config
