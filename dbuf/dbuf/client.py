import asyncio
import aiohttp
import collections
import struct

from hailtop.config import get_deploy_config

from .logging import log
from .retry_forever import retry_aiohttp_forever


def grouped(xs, size):
    n = len(xs)
    i = 0
    while i < n:
        yield xs[i:(i+size)]
        i += size


class DBufClient:
    async def __aenter__(self):
        self.aiosession = aiohttp.ClientSession(raise_for_status=True,
                                                timeout=aiohttp.ClientTimeout(total=60))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aiosession.close()

    def __init__(self,
                 name,
                 id=None,
                 max_bufsize=10*1024*1024 - 1,
                 retry_delay=1,
                 deploy_config=None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        self.deploy_config = deploy_config
        self.root_url = deploy_config.base_url(name)
        self.session_url = None if id is None else f'{self.root_url}/s/{id}'
        self.aiosession = None
        self.id = id
        self.buf = bytearray(max_bufsize)
        self.offs = []
        self.sizes = []
        self.cursor = 0
        self.max_bufsize = max_bufsize
        self.retry_delay = retry_delay

    async def create(self):
        async with self.aiosession.post(f'{self.root_url}/s') as resp:
            assert resp.status == 200
            self.id = int(await resp.text())
        self.session_url = f'{self.root_url}/s/{self.id}'
        return self.id

    async def append(self, data):
        n = len(data)
        assert n < self.max_bufsize
        keys = []
        if self.cursor + n > self.max_bufsize:
            keys = await self.flush()
        self.buf[self.cursor:(self.cursor+n)] = data
        self.offs.append(self.cursor)
        self.sizes.append(n)
        self.cursor += n
        return keys

    async def flush(self):
        if self.cursor == 0:
            return []
        retry_delay = self.retry_delay
        while True:
            try:
                buf = self.buf
                offs = self.offs
                sizes = self.sizes
                cursor = self.cursor
                self.buf = bytearray(self.max_bufsize)
                self.offs = []
                self.sizes = []
                self.cursor = 0
                async with self.aiosession.post(self.session_url, data=buf[0:cursor]) as resp:
                    assert resp.status == 200
                    server, file_id, pos, _ = await resp.json()
                    return [(server, file_id, pos + off, size)
                            for off, size in zip(offs, sizes)]
            except (aiohttp.client_exceptions.ClientOSError,
                    aiohttp.client_exceptions.ClientConnectorError) as exc:

                log.info(f'backing off due to {exc}')
                await asyncio.sleep(retry_delay)
                retry_delay = retry_delay * 2

    async def get(self, key, retry_delay=1):
        server = key[0]
        while True:
            try:
                async with self.aiosession.post(f'{server}/s/{self.id}/get', json=key) as resp:
                    assert resp.status == 200
                    return await resp.read()
            except (aiohttp.client_exceptions.ClientResponseError,
                    aiohttp.client_exceptions.ClientOSError,
                    aiohttp.client_exceptions.ClientConnectorError) as exc:
                log.warning(f'backing off due to {exc}')
                await asyncio.sleep(retry_delay)
                retry_delay = retry_delay * 2

    def decode(self, byte_array):
        off = 0
        result = []
        while off < len(byte_array):
            n2 = struct.unpack_from("i", byte_array, off)[0]
            off += 4
            result.append(byte_array[off:(off+n2)])
            off += n2
        return result

    async def getmany(self, keys, retry_delay=1):
        servers = collections.defaultdict(list)
        results = [None for _ in keys]
        for i, key in enumerate(keys):
            servers[key[0]].append((key, i))
        for server, keys in servers.items():
            i = 0
            while i < len(keys):
                batch = []
                size = 0
                while i < len(keys):
                    assert keys[i][0][3] < self.max_bufsize
                    if size + keys[i][0][3] < self.max_bufsize:
                        batch.append(keys[i])
                        size += keys[i][0][3]
                        i += 1
                    else:
                        break

                async def http():
                    async with self.aiosession.post(f'{server}/s/{self.id}/getmany',
                                                    json=[x[0] for x in batch]) as resp:
                        assert resp.status == 200
                        data = await resp.read()
                        for v, j in zip(self.decode(data), (x[1] for x in batch)):
                            results[j] = v
                await retry_aiohttp_forever(http)
        return results

    async def delete(self):
        async with self.aiosession.delete(self.session_url) as resp:
            assert resp.status == 200

    async def get_workers(self):
        async with self.aiosession.get(f'{self.root_url}/w') as resp:
            assert resp.status == 200
            return await resp.json()
