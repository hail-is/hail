import random
import aiohttp
import collections
import struct

import hailtop.utils as utils
from hailtop.config import get_deploy_config


class DBufClient:
    async def __aenter__(self):
        self.aiosession = aiohttp.ClientSession(raise_for_status=True,
                                                timeout=aiohttp.ClientTimeout(total=60))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aiosession.close()

    def __init__(self,
                 leader_name,
                 id=None,
                 max_bufsize=10 * 1024 * 1024 - 1,
                 deploy_config=None,
                 rng=None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.deploy_config = deploy_config
        self.root_url = deploy_config.base_url(leader_name)
        self.session_url = None if id is None else f'{self.root_url}/s/{id}'
        self.aiosession = None
        self.id = id
        self.offs = []
        self.sizes = []
        self.cursor = 0
        self.max_bufsize = max_bufsize

    async def create(self):
        resp = await utils.request_retry_transient_errors(
            self.aiosession,
            'POST',
            f'{self.root_url}/s')
        self.id = int(await resp.text())
        self.session_url = f'{self.root_url}/s/{self.id}'
        return self.id

    async def start_write(self):
        workers = await self.get_workers()
        worker = workers[self.rng.randrange(len(workers))]
        return DBufAppender(worker, self.deploy_config, self.id, self.max_bufsize, self.aiosession)

    async def get(self, key):
        server = key[0]
        server_url = self.deploy_config.base_url(server)

        resp = await utils.request_retry_transient_errors(
            self.aiosession,
            'POST',
            f'{server_url}/s/{self.id}/get',
            json=key)
        return await resp.read()

    def _decode(self, byte_array):
        off = 0
        result = []
        while off < len(byte_array):
            n2 = struct.unpack_from("i", byte_array, off)[0]
            off += 4
            result.append(byte_array[off:(off + n2)])
            off += n2
        return result

    async def getmany(self, keys):
        servers = collections.defaultdict(list)
        results = [None for _ in keys]
        for i, key in enumerate(keys):
            servers[key[0]].append((key, i))

        def get_from_server(server, keys):
            async def f():
                server_url = self.deploy_config.base_url(server)
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

                    resp = await utils.request_retry_transient_errors(
                        self.aiosession,
                        'POST',
                        f'{server_url}/s/{self.id}/getmany',
                        json=[x[0] for x in batch])
                    data = await resp.read()
                    for v, j in zip(self._decode(data), (x[1] for x in batch)):
                        results[j] = v
            return f
        await utils.bounded_gather(*[get_from_server(server, keys)
                                     for server, keys in servers.items()])
        assert all(x is not None for x in results), results
        return results

    async def delete(self):
        await utils.request_retry_transient_errors(
            self.aiosession, 'DELETE', self.session_url)

    async def get_workers(self):
        return await (await utils.request_retry_transient_errors(
            self.aiosession, 'GET', f'{self.root_url}/w')).json()


class DBufAppender:
    def __init__(self, worker_name, deploy_config, id, max_bufsize, aiosession):
        self.aiosession = aiosession
        server_url = deploy_config.base_url(worker_name)
        self.session_url = f'{server_url}/s/{id}'
        self.buf = bytearray(max_bufsize)
        self.offs = []
        self.sizes = []
        self.cursor = 0
        self.keys = []

    async def write(self, data):
        n = len(data)
        assert n < len(self.buf)
        if self.cursor + n > len(self.buf):
            await self.flush()
        self.buf[self.cursor:(self.cursor + n)] = data
        self.offs.append(self.cursor)
        self.sizes.append(n)
        self.cursor += n

    async def flush(self):
        if self.cursor == 0:
            return
        buf = self.buf
        offs = self.offs
        sizes = self.sizes
        cursor = self.cursor
        n_keys = len(self.keys)
        key_range = slice(n_keys, n_keys + len(self.offs))
        self.keys.extend(None for _ in self.offs)
        self.buf = bytearray(len(self.buf))
        self.offs = []
        self.sizes = []
        self.cursor = 0

        resp = await utils.request_retry_transient_errors(
            self.aiosession,
            'POST',
            self.session_url,
            data=buf[0:cursor])
        server, file_id, pos, _ = await resp.json()
        self.keys[key_range] = [(server, file_id, pos + off, size)
                                for off, size in zip(offs, sizes)]

    async def finish(self):
        await self.flush()
        self.buf = None
        return self.keys
