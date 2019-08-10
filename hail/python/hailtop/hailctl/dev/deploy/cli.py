import asyncio
import aiohttp

from hailtop.gear.auth.hailjwt import find_token

profiles = {
    "batch_test": [
        'default_ns',
        'deploy_batch_sa',
        'batch_pods_ns',
        'deploy_batch_output_sa',
        'base_image',
        'create_accounts',
        'batch_image',
        'batch_database',
        'create_batch_tables_image',
        'create_batch_tables',
        'create_batch_tables2',
        'deploy_batch',
        'deploy_batch_pods',
        'test_batch_image',
        'test_batch'
    ]
}

def init_parser(parser):
    parser.add_argument('github_username', type=str)
    parser.add_argument('repo', type=str)
    parser.add_argument('branch', type=str)
    parser.add_argument('namespace', type=str)
    parser.add_argument('profile', type=str, choices=list(profiles.keys()))

def main(args):
    asyncio.run(submit(args))

async def submit(args):
    token = find_token()
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        data = {
            'userdata': {
                'username': args.github_username
            },
            'repo': args.repo,
            'branch': args.branch,
            'profile': profiles[args.profile],
            'namespace': args.namespace
        }
        print(f"Submitting: {data}")
        async with session.post('https://ci.hail.is/api/v1alpha/dev_test_branch/', json=data) as resp:
            if (resp.status == 200):
                text = await resp.text()
                print(f"Created batch {text}")
            else:
                print(f"Error: Returned status code {resp.status}")
