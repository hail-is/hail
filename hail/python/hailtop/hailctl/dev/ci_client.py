import aiohttp
import sys

from typing import Optional

from hailtop import httpx
from hailtop.config import get_deploy_config
from hailtop.auth import hail_credentials
from hailtop.httpx import client_session


class CIClient:
    def __init__(self, deploy_config=None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        self._deploy_config = deploy_config
        self._session: Optional[httpx.ClientSession] = None

    async def __aenter__(self):
        async with hail_credentials() as credentials:
            headers = await credentials.auth_headers()
        self._session = client_session(raise_for_status=False, timeout=aiohttp.ClientTimeout(total=60), headers=headers)  # type: ignore
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def dev_deploy_branch(self, branch, steps, excluded_steps, extra_config):
        data = {
            'branch': branch,
            'steps': steps,
            'excluded_steps': excluded_steps,
            'extra_config': extra_config,
        }
        assert self._session
        async with self._session.post(
            self._deploy_config.url('ci', '/api/v1alpha/dev_deploy_branch'), json=data
        ) as resp:
            if resp.status >= 400:
                print(f'HTTP Response code was {resp.status}')
                print(await resp.text())
                sys.exit(1)
            resp_data = await resp.json()
            return resp_data['batch_id']
