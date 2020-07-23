import asyncio
import webbrowser
import aiohttp
import sys

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.tls import get_context_specific_ssl_client_session


def init_parser(parser):
    parser.add_argument("--branch", "-b", type=str,
                        help="Fully-qualified branch, e.g., hail-is/hail:feature.", required=True)
    parser.add_argument("--steps", "-s", type=str,
                        help="Comma-separated list of steps to run.", required=True)
    parser.add_argument("--open", "-o",
                        action="store_true",
                        help="Open the deploy batch page in a web browser.")


class CIClient:
    def __init__(self, deploy_config=None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        self._deploy_config = deploy_config
        self._session = None

    async def __aenter__(self):
        headers = service_auth_headers(self._deploy_config, 'ci')
        self._session = get_context_specific_ssl_client_session(
            timeout=aiohttp.ClientTimeout(total=60), headers=headers)
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


async def submit(args):
    deploy_config = get_deploy_config()
    steps = args.steps.split(',')
    steps = [s.strip() for s in steps]
    steps = [s for s in steps if s]
    async with CIClient(deploy_config) as ci_client:
        batch_id = await ci_client.dev_deploy_branch(args.branch, steps)
        url = deploy_config.url('ci', f'/batches/{batch_id}')
        print(f'Created deploy batch, see {url}')
        if args.open:
            webbrowser.open(url)


def main(args):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(submit(args))
    loop.run_until_complete(loop.shutdown_asyncgens())
