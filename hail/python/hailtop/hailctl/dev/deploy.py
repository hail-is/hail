import asyncio
import webbrowser
import aiohttp
import sys
import click

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.httpx import client_session

from ..dev import dev


class CIClient:
    def __init__(self, deploy_config=None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        self._deploy_config = deploy_config
        self._session = None

    async def __aenter__(self):
        headers = service_auth_headers(self._deploy_config, 'ci')
        self._session = client_session(
            timeout=aiohttp.ClientTimeout(total=5), headers=headers)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def dev_deploy_branch(self, branch, steps):
        data = {
            'branch': branch,
            'steps': steps
        }
        async with self._session.post(
                self._deploy_config.url('ci', '/api/v1alpha/dev_deploy_branch'), json=data) as resp:
            if resp.status >= 400:
                print(f'HTTP Response code was {resp.status}')
                print(await resp.text())
                sys.exit(1)
            resp_data = await resp.json()
            return resp_data['batch_id']


async def submit(branch, steps, open):
    deploy_config = get_deploy_config()

    steps = [s.strip()
             for step in steps
             for s in step.split(',') if s.strip()]
    async with CIClient(deploy_config) as ci_client:
        batch_id = await ci_client.dev_deploy_branch(branch, steps)
        url = deploy_config.url('ci', f'/batches/{batch_id}')
        print(f'Created deploy batch, see {url}')
        if open:
            webbrowser.open(url)


@dev.command(
    help="Deploy a branch.")
@click.option("--branch", "-b",
              required=True,
              help="Fully-qualified branch, e.g., hail-is/hail:feature.")
@click.option('--steps', '-s',
              required=True, multiple=True,
              help="Comma list of steps to run.")
@click.option("--open", "-o", is_flag=True,
              help="Open the deploy batch page in a web browser.")
def deploy(branch, steps, open):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(submit(branch, steps, open))
    loop.run_until_complete(loop.shutdown_asyncgens())
