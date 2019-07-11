import aiohttp
import asyncio
import ci

def init_parser(parser):
    parser.add_argument('repo', type=str)
    parser.add_argument('branch', type=str)
    parser.add_argument('namespace', type=str)
    parser.add_argument('profile', type=str, choices=['batch_test'])

def main(args):
    # For now, let's find CI and just manually call the damn function.
    async with aiohttp.ClientSession() as session:
        session.post('https://batch.hail.is/api/v1alpha/dev_test_branch/'', json={
            'userdata': {
                'username': 'johnc1231'
            },
            'repo': args.repo,
            'branch': args.branch,
            'profile': args.profile,
            'namespace': args.namespace
        })

