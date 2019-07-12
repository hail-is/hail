import asyncio
import aiohttp

def init_parser(parser):
    parser.add_argument('repo', type=str)
    parser.add_argument('branch', type=str)
    parser.add_argument('namespace', type=str)
    parser.add_argument('profile', type=str, choices=['batch_test'])

def main(args):
    asyncio.run(submit(args))

async def submit(args):
    async with aiohttp.ClientSession() as session:
        data = {
            'userdata': {
                'username': 'johnc1231'
            },
            'repo': args.repo,
            'branch': args.branch,
            'profile': args.profile,
            'namespace': args.namespace
        }
        async with session.post('https://ci.hail.is/api/v1alpha/dev_test_branch/', json=data) as resp:
            print(await resp.text())
