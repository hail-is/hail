import asyncio
import webbrowser
import aiohttp

from hailtop.gear import get_deploy_config
from hailtop.gear.auth import auth_headers


def init_parser(parser):
    parser.add_argument("--branch", "-b", type=str,
                        help="Fully-qualified branch, e.g., hail-is/hail:feature.", required=True)
    parser.add_argument("--steps", "-s", type=str,
                        help="Comma-separated list of steps to run.", required=True)
    parser.add_argument("--open", "-o",
                        action="store_true",
                        help="Open the deploy batch page in a web browser.")


class CIClient:
    def __init__(self):
        self._session = None

    async def __aenter__(self):
        headers = auth_headers('ci')
        self._session = aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60), headers=headers)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def dev_deploy_branch(self, branch, steps):
        deploy_config = get_deploy_config()
        data = {
            'branch': branch,
            'steps': steps
        }
        async with self._session.post(
                deploy_config.url('ci', '/api/v1alpha/dev_deploy_branch'), json=data) as resp:
            resp_data = await resp.json()
            return resp_data['batch_id']


async def submit(args):
    steps = args.steps.split(',')
    steps = [s.strip() for s in steps]
    steps = [s for s in steps if s]
    async with CIClient() as ci_client:
        batch_id = await ci_client.dev_deploy_branch(args.branch, steps)
        deploy_config = get_deploy_config()
        url = deploy_config.url('ci', f'/batches/{batch_id}')
        print(f'Created deploy batch, see {url}')
        if args.open:
            webbrowser.open(url)


def main(args):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(submit(args))
    loop.run_until_complete(loop.shutdown_asyncgens())
