import json
import logging
import asyncio
import pytest
import aiohttp

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.httpx import client_session
import hailtop.utils as utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

sha = 'd626f793ad700c45a878d192652a0378818bbd8b'


@pytest.mark.asyncio
async def test_update_commits():
    deploy_config = get_deploy_config()
    headers = service_auth_headers(deploy_config, 'benchmark')
    commit_benchmark_url = deploy_config.url('benchmark', f'/api/v1alpha/benchmark/commit/{sha}')

    async def request(method):
        return await utils.request_retry_transient_errors(
            session, method, f'{commit_benchmark_url}', headers=headers, json={'sha': sha}
        )

    async with client_session() as session:
        await request('DELETE')

        resp = await request('GET')
        commit = await resp.json()
        assert commit['status'] is None, commit

        resp = await request('POST')
        commit = await resp.json()

        while commit['status'] is not None and not commit['status']['complete']:
            await asyncio.sleep(5)
            resp = await request('GET')
            commit = await resp.json()
            print(commit['status'])
