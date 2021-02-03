import asyncio
import sys

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.httpx import client_session


def init_parser(parser):
    subparsers = parser.add_subparsers()
    set_parser = subparsers.add_parser(
        'set',
        help='Set a Hail query resource value.',
        description='Set a Hail query resource value.')
    unset_parser = subparsers.add_parser(
        'unset',
        help='Unset a Hail query resource value (restore to default behavior).',
        description='Unset a Hail query resource value (restore to default behavior).')
    get_parser = subparsers.add_parser(
        'get',
        help='Get the value of a Hail query resource (or all values of a specific resource type).',
        description='Get the value of a Hail query resource.')

    resource_types = ['flag']

    set_parser.set_defaults(action='set')
    set_parser.add_argument("resource", type=str, choices=resource_types,
                            help="Resource type.")
    set_parser.add_argument("name", type=str,
                            help="Name of resource.")
    set_parser.add_argument("value", type=str,
                            help="Value to set.")

    unset_parser.set_defaults(action='unset')
    unset_parser.add_argument("resource", type=str, choices=resource_types,
                              help="Resource type.")
    unset_parser.add_argument("name", type=str,
                              help="Name of resource.")

    get_parser.set_defaults(action='get')
    get_parser.add_argument("resource", type=str, choices=resource_types,
                            help="Resource type.")
    get_parser.add_argument("name", type=str, nargs='*',
                            help="Name(s) of resource.")


class QueryClient:
    def __init__(self):
        self._deploy_config = get_deploy_config()
        self._session = None

    async def __aenter__(self):
        headers = service_auth_headers(self._deploy_config, 'query')
        self._session = client_session(
            raise_for_status=False,
            headers=headers)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def get_request(self, path, params=None, handler=None):
        async with self._session.get(
                self._deploy_config.url('query', f'/api/v1alpha/{path}'), params=params) as resp:
            if resp.status >= 400:
                if handler is not None:
                    handler(resp.status, await resp.text())
                print(f'HTTP Response code was {resp.status}')
                print(await resp.text())
                sys.exit(1)
            return await resp.json()

    async def set_flag(self, name, value):
        params = {'value': value} if value is not None else None
        return await self.get_request(f'flags/set/{name}', params=params)

    async def get_flag(self, names):
        if len(names) == 0:
            all = await self.get_request('flags/get')
            return all, []
        flags = {}
        invalid = []
        for name in names:
            try:
                def raise_flag_not_found(status, text):
                    error = text.split('\n')[0].strip()
                    if status == 500 and 'java.util.NoSuchElementException: key not found: ' in error:
                        raise KeyError(error[49:])
                flags[name] = await self.get_request(f'flags/get/{name}', handler=raise_flag_not_found)
            except KeyError as e:
                invalid.append(str(e))
        return flags, invalid


async def submit(args):
    async with QueryClient() as client:
        if args.action == 'set' and args.resource == 'flag':
            old = await client.set_flag(args.name, args.value)
            print(f'Set {args.name} to {args.value}. Old value: {old}')
        elif args.action == 'unset' and args.resource == 'flag':
            old = await client.set_flag(args.name, None)
            print(f'Unset {args.name}. Old value: {old}')
        elif args.action == 'get' and args.resource == 'flag':
            flags, invalid = await client.get_flag(args.name)
            n = max(len(k) for k in flags.keys())
            for k, v in flags.items():
                print(f'{k.rjust(n)}: ' + ('null' if v is None else f'"{v}"'))
            if len(invalid) > 0:
                print("Invalid keys: ")
                for i in invalid:
                    print(f'  {i}')


def main(args):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(submit(args))
    loop.run_until_complete(loop.shutdown_asyncgens())
