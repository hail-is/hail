import asyncio
import aiohttp
from asyncinit import asyncinit

from hailtop.gear.auth.hailjwt import find_token

def init_parser(parser):
    parser.add_argument("--branch", "-b", type=str,
                        help="Fully-qualified branch, e.g., hail-is/hail:feature.")
    parser.add_argument("--steps", "-s", type=str,
                        help="Comma-separated list of steps to run.")

@asyncinit
class CIClient:
    async def __init__(self):
        token = find_token()
        self._session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"Authorization": f"Bearer {token}"})

    async def __aenter__(self):
        pass

    async def __aexit__(self):
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
                'https://ci.hail.is/api/v1alpha/dev_test_branch/', json=data) as resp:
            resp_data = resp.json()
            return resp_data['batch_id']

    
async def submit(args):
    steps = args.steps.split(',')
    steps = [s.strip() for s in steps]
    steps = [s for s in steps if s]
    async with CIClient() as ci_client:
        batch_id = ci_client.dev_deploy_branch(args.branch, steps)
        print(f'Created deploy batch, see https://ci.hail.is/batches/{batch_id}')


def main(args):
    asyncio.run(submit(args))
