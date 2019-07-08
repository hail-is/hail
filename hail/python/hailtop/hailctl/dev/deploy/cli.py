import aiohttp
import ci

def init_parser(parser):
    parser.add_argument(name='repo', type=str)
    parser.add_argument(name='branch', type=str)
    parser.add_argument(name='profile', type=str, choices=['batch_test'])

def main(args):
    # For now, let's find CI and just manually call the damn function. 
    ci.dev_test_branch()