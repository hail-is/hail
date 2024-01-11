import base64
import os
import tempfile
from typing import Dict, List

import orjson
from aiohttp import web

from hailtop import httpx
from hailtop.aiocloud import aiogoogle
from hailtop.auth.auth import IdentityProvider
from hailtop.utils import check_exec_output

from ....globals import HTTP_CLIENT_MAX_SIZE
from ....worker.worker_api import CloudWorkerAPI, ContainerRegistryCredentials, HailMetadataServer
from ..instance_config import GCPSlimInstanceConfig
from .disk import GCPDisk


class GCPWorkerAPI(CloudWorkerAPI):
    nameserver_ip = '169.254.169.254'

    # async because GoogleSession must be created inside a running event loop
    @staticmethod
    async def from_env() -> 'GCPWorkerAPI':
        project = os.environ['PROJECT']
        zone = os.environ['ZONE'].rsplit('/', 1)[1]
        worker_credentials = aiogoogle.GoogleInstanceMetadataCredentials()
        http_session = httpx.ClientSession()
        return GCPWorkerAPI(project, zone, worker_credentials, http_session)

    def __init__(
        self,
        project: str,
        zone: str,
        worker_credentials: aiogoogle.GoogleInstanceMetadataCredentials,
        http_session: httpx.ClientSession,
    ):
        self.project = project
        self.zone = zone
        self._http_session = http_session
        self._compute_client = aiogoogle.GoogleComputeClient(project)
        self._gcsfuse_credential_files: Dict[str, str] = {}
        self._worker_credentials = worker_credentials

    @property
    def cloud_specific_env_vars_for_user_jobs(self) -> List[str]:
        idp_json = orjson.dumps({'idp': IdentityProvider.GOOGLE.value}).decode('utf-8')
        return [
            'GOOGLE_APPLICATION_CREDENTIALS=/gsa-key/key.json',
            f'HAIL_IDENTITY_PROVIDER_JSON={idp_json}',
        ]

    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> GCPDisk:
        return GCPDisk(
            zone=self.zone,
            project=self.project,
            instance_name=instance_name,
            name=disk_name,
            size_in_gb=size_in_gb,
            mount_path=mount_path,
            compute_client=self._compute_client,
        )

    async def worker_container_registry_credentials(self, session: httpx.ClientSession) -> ContainerRegistryCredentials:
        access_token = await self._worker_credentials.access_token()
        return {'username': 'oauth2accesstoken', 'password': access_token}

    async def user_container_registry_credentials(self, credentials: Dict[str, str]) -> ContainerRegistryCredentials:
        key = orjson.loads(base64.b64decode(credentials['key.json']).decode())
        async with aiogoogle.GoogleServiceAccountCredentials(key) as sa_credentials:
            access_token = await sa_credentials.access_token()
        return {'username': 'oauth2accesstoken', 'password': access_token}

    def metadata_server(self) -> 'GoogleHailMetadataServer':
        return GoogleHailMetadataServer(self.project, self._http_session)

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> GCPSlimInstanceConfig:
        return GCPSlimInstanceConfig.from_dict(config_dict)

    def _write_gcsfuse_credentials(self, credentials: Dict[str, str], mount_base_path_data: str) -> str:
        if mount_base_path_data not in self._gcsfuse_credential_files:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as credsfile:
                credsfile.write(base64.b64decode(credentials['key.json']).decode())
                self._gcsfuse_credential_files[mount_base_path_data] = credsfile.name
        return self._gcsfuse_credential_files[mount_base_path_data]

    async def _mount_cloudfuse(
        self,
        credentials: Dict[str, str],
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):  # pylint: disable=unused-argument

        fuse_credentials_path = self._write_gcsfuse_credentials(credentials, mount_base_path_data)

        bucket = config['bucket']
        assert bucket

        options = ['allow_other']
        if config['read_only']:
            options.append('ro')

        try:
            billing_project_flag = ['--billing-project', config["requester_pays_project"]]
        except KeyError:
            billing_project_flag = []

        await check_exec_output(
            '/usr/bin/gcsfuse',
            '-o',
            ','.join(options),
            '--file-mode',
            '770',
            '--dir-mode',
            '770',
            '--implicit-dirs',
            '--key-file',
            fuse_credentials_path,
            *billing_project_flag,
            bucket,
            mount_base_path_data,
        )

    async def unmount_cloudfuse(self, mount_base_path_data: str):
        try:
            await check_exec_output('fusermount', '-u', mount_base_path_data)
        finally:
            os.remove(self._gcsfuse_credential_files[mount_base_path_data])
            del self._gcsfuse_credential_files[mount_base_path_data]

    async def close(self):
        await self._compute_client.close()

    def __str__(self):
        return f'project={self.project} zone={self.zone}'


class GoogleHailMetadataServer(HailMetadataServer):
    def __init__(self, project: str, http_session: httpx.ClientSession):
        super().__init__()
        self._project = project
        self._metadata_server_client = aiogoogle.GoogleMetadataServerClient(http_session)
        self._ip_container_credentials: Dict[str, aiogoogle.GoogleServiceAccountCredentials] = {}

    def set_container_credentials(self, ip: str, credentials: Dict[str, str]):
        key = orjson.loads(base64.b64decode(credentials['key.json']).decode())
        self._ip_container_credentials[ip] = aiogoogle.GoogleServiceAccountCredentials(key)

    async def clear_container_credentials(self, ip: str):
        await self._ip_container_credentials.pop(ip).close()

    def _container_credentials(self, request: web.Request) -> Dict[str, aiogoogle.GoogleServiceAccountCredentials]:
        assert request.remote
        if request.remote not in self._ip_container_credentials:
            raise web.HTTPBadRequest()
        credentials = self._ip_container_credentials[request.remote]
        return {'default': credentials, credentials.email: credentials}

    def _user_credentials(self, request: web.Request) -> aiogoogle.GoogleServiceAccountCredentials:
        email = request.match_info.get('gsa') or 'default'
        return self._container_credentials(request)[email]

    async def root(self, _):
        return web.Response(text='computeMetadata/\n')

    async def project_id(self, _):
        return web.Response(text=self._project)

    async def numeric_project_id(self, _):
        return web.Response(text=await self._metadata_server_client.numeric_project_id())

    async def service_accounts(self, request: web.Request):
        accounts = '\n'.join(self._container_credentials(request).keys())
        return web.Response(text=f'{accounts}\n')

    async def user_service_account(self, request: web.Request):
        gsa_email = self._user_credentials(request).email
        recursive = request.query.get('recursive')
        # https://cloud.google.com/compute/docs/metadata/querying-metadata
        # token is not included in the recursive version, presumably as that
        # is not simple metadata but requires requesting an access token
        if recursive == 'true':
            return web.json_response(
                {
                    'aliases': ['default'],
                    'email': gsa_email,
                    'scopes': ['https://www.googleapis.com/auth/cloud-platform'],
                },
            )
        return web.Response(text='aliases\nemail\nscopes\ntoken\n')

    async def user_email(self, request: web.Request):
        return web.Response(text=self._user_credentials(request).email)

    async def user_token(self, request: web.Request):
        gsa_email = request.match_info['gsa']
        creds = self._container_credentials(request)[gsa_email]
        access_token = await creds._get_access_token()
        return web.json_response(
            {
                'access_token': access_token.token,
                'expires_in': access_token.expires_in,
                'token_type': 'Bearer',
            }
        )

    @web.middleware
    async def configure_response(self, request: web.Request, handler):
        credentials = self._container_credentials(request)
        gsa = request.match_info.get('gsa', 'default')
        if gsa not in credentials:
            raise web.HTTPBadRequest()

        response = await handler(request)
        response.enable_compression()

        # `gcloud` does not properly respect `charset`, which aiohttp automatically
        # sets so we have to explicitly erase it
        # See https://github.com/googleapis/google-auth-library-python/blob/b935298aaf4ea5867b5778bcbfc42408ba4ec02c/google/auth/compute_engine/_metadata.py#L170
        if 'application/json' in response.headers['Content-Type']:
            response.headers['Content-Type'] = 'application/json'
        response.headers['Metadata-Flavor'] = 'Google'
        response.headers['Server'] = 'Metadata Server for VM'
        response.headers['X-XSS-Protection'] = '0'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        return response

    def create_app(self) -> web.Application:
        metadata_app = web.Application(
            client_max_size=HTTP_CLIENT_MAX_SIZE,
            middlewares=[self.configure_response],
        )
        metadata_app.add_routes(
            [
                web.get('/', self.root),
                web.get('/computeMetadata/v1/project/project-id', self.project_id),
                web.get('/computeMetadata/v1/project/numeric-project-id', self.numeric_project_id),
                web.get('/computeMetadata/v1/instance/service-accounts/', self.service_accounts),
                web.get('/computeMetadata/v1/instance/service-accounts/{gsa}/', self.user_service_account),
                web.get('/computeMetadata/v1/instance/service-accounts/{gsa}/email', self.user_email),
                web.get('/computeMetadata/v1/instance/service-accounts/{gsa}/token', self.user_token),
            ]
        )
        return metadata_app
