import os
from typing import Dict

import aiohttp

from hailtop import httpx
from hailtop.aiocloud import aiogoogle
from hailtop.utils import retry_transient_errors

from ....worker.worker_api import CloudWorkerAPI
from ..instance_config import GCPSlimInstanceConfig
from .credentials import GCPUserCredentials
from .disk import GCPDisk


class GCPWorkerAPI(CloudWorkerAPI):
    nameserver_ip = '169.254.169.254'

    # async because GoogleSession must be created inside a running event loop
    @staticmethod
    async def from_env() -> 'GCPWorkerAPI':
        project = os.environ['PROJECT']
        zone = os.environ['ZONE'].rsplit('/', 1)[1]
        session = aiogoogle.GoogleSession()
        return GCPWorkerAPI(project, zone, session)

    def __init__(self, project: str, zone: str, session: aiogoogle.GoogleSession):
        self.project = project
        self.zone = zone
        self._google_session = session
        self._compute_client = aiogoogle.GoogleComputeClient(project, session=session)

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

    def get_cloud_async_fs(self) -> aiogoogle.GoogleStorageAsyncFS:
        return aiogoogle.GoogleStorageAsyncFS(session=self._google_session)

    def get_compute_client(self) -> aiogoogle.GoogleComputeClient:
        return self._compute_client

    def user_credentials(self, credentials: Dict[str, str]) -> GCPUserCredentials:
        return GCPUserCredentials(credentials)

    async def worker_access_token(self, session: httpx.ClientSession) -> Dict[str, str]:
        token_dict = await retry_transient_errors(
            session.post_return_json,
            'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'},
            timeout=aiohttp.ClientTimeout(total=60),  # type: ignore
        )
        access_token = token_dict['access_token']
        return {'username': 'oauth2accesstoken', 'password': access_token}

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> GCPSlimInstanceConfig:
        return GCPSlimInstanceConfig.from_dict(config_dict)

    def write_cloudfuse_credentials(
        self, root_dir: str, credentials: str, bucket: str
    ) -> str:  # pylint: disable=unused-argument
        path = f'{root_dir}/cloudfuse/key.json'
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path))
            with open(path, 'w', encoding='utf-8') as f:
                f.write(credentials)
        return path

    def _mount_cloudfuse(
        self, fuse_credentials_path: str, mount_base_path_data: str, mount_base_path_tmp: str, config: dict
    ) -> str:  # pylint: disable=unused-argument
        bucket = config['bucket']
        assert bucket

        options = ['allow_other']
        if config['read_only']:
            options.append('ro')

        try:
            billing_project_flag = f'--billing-project "{config["requester_pays_project"]}"'
        except KeyError:
            billing_project_flag = ''

        return f'''
/usr/bin/gcsfuse \
    -o {",".join(options)} \
    --file-mode 770 \
    --dir-mode 770 \
    --implicit-dirs \
    --key-file {fuse_credentials_path} \
    {billing_project_flag} \
    {bucket} {mount_base_path_data}
'''

    def _unmount_cloudfuse(self, mount_base_path: str) -> str:
        return f'fusermount -u {mount_base_path}'

    def __str__(self):
        return f'project={self.project} zone={self.zone}'
