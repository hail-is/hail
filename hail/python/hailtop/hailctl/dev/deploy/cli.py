import aiohttp
import asyncio
import ci

def init_parser(parser):
    parser.add_argument('repo', type=str)
    parser.add_argument('branch', type=str)
    parser.add_argument('profile', type=str, choices=['batch_test'])

def main(args):
    # For now, let's find CI and just manually call the damn function. 
    asyncio.run(ci.dev_test_branch({
        'userdata': {
            'username': 'johnc1231'
        },
        'repo': args.repo,
        'branch': args.branch,
        'profile': args.profile
    }))