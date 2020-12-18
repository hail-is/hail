import asyncio
import aiohttp
import sys
import click

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.httpx import client_session

from ..dev import dev


class QueryClient:
    def __init__(self):
        self._deploy_config = get_deploy_config()
        self._session = None

    async def __aenter__(self):
        headers = service_auth_headers(self._deploy_config, 'query')
        self._session = client_session(
            raise_for_status=False,
            timeout=aiohttp.ClientTimeout(total=5), headers=headers)
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


def async_run(coro):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(coro)
    loop.run_until_complete(loop.shutdown_asyncgens())


@dev.group()
def query():
    pass


@query.command(
    help="Set a Hail Query resource value.")
@click.argument('name')
@click.argument('value')
def set(name, value):
    async def async_main(name, value):
        async with QueryClient() as client:
            old = await client.set_flag(name, value)
            print(f"Set {name} to {value}. Old value: {old}.")

    async_run(async_main(name, value))


@query.command(
    help="Unset a Hail Query resource value (restore to default behavior).")
@click.argument('name')
def unset(name):
    async def async_main(name):
        async with QueryClient() as client:
            old = await client.set_flag(name, None)
            print(f'Unset {name}. Old value: {old}.')

    async_run(async_main(name))


@query.command(
    help="Get the value of a Hail Query resource.")
@click.argument('name')
def get(name):
    async def async_main(name):
        async with QueryClient() as client:
            flags, invalid = await client.get_flag(name)
        n = max(len(k) for k in flags.keys())
        for k, v in flags.items():
            print(f'{k.rjust(n)}: ' + ('null' if v is None else f'"{v}"'))
        if len(invalid) > 0:
            print("Invalid keys: ")
            for i in invalid:
                print(f'  {i}')

    async_run(async_main(name))
