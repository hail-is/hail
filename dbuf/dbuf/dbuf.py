import aiohttp
import aiohttp.web as web
import argparse
import asyncio
import os
import shutil
import struct

import hailtop.utils as utils
from hailtop.config import get_deploy_config
from gear import AccessLogger

from . import aiofiles as af
from .logging import log


class Session:
    @staticmethod
    async def make(bufsize, data_dir, id, aiofiles):
        await aiofiles.mkdir(f'{data_dir}/{id}')
        return Session(bufsize, data_dir, id, aiofiles)

    def __init__(self, bufsize, data_dir, id, aiofiles):
        self.data_dir = data_dir
        self.id = id
        self.aiofiles = aiofiles
        self.next_file = 1
        self.current_file = 0
        self.max_size = bufsize
        self.buf = bytearray(self.max_size)
        self.bufs = [bytearray(self.max_size)]
        self.cursor = 0
        self.writers = dict()
        self.file_budget = asyncio.BoundedSemaphore(128)

    async def _flush_to_file(self, fname, buf, cursor):
        async with self.file_budget:
            with await self.aiofiles.open(fname, 'wb', buffering=0) as f:
                log.warning(f'open spilling to {fname}')
                written = await self.aiofiles.write(f, buf[0:cursor])
                assert written == cursor
                self.bufs.append(buf)
                log.warning(f'done spilling to {fname}')

    async def write(self, data):
        n = len(data)
        assert n < self.max_size
        if self.cursor + n > self.max_size:
            written_file_id = self.current_file
            buf = self.buf
            cursor = self.cursor
            if len(self.bufs) > 0:
                self.buf = self.bufs.pop()
            else:
                self.buf = bytearray(self.max_size)
            self.cursor = 0
            self.current_file = self.next_file
            self.next_file += 1

            fname = f'{self.data_dir}/{self.id}/{written_file_id}'
            self.writers[written_file_id] = asyncio.ensure_future(self._flush_to_file(fname, buf, cursor))
        pos = self.cursor
        self.buf[pos:(pos+n)] = data
        self.cursor += n
        return self.current_file, pos, n

    async def read(self, key, buf):
        file_id, pos, n = key
        if file_id == self.current_file:
            buf[0:n] = self.buf[pos:(pos+n)]
            return
        writer = self.writers.get(file_id)
        if writer is not None:
            await writer
            assert not writer.cancelled()
            assert writer.exception() is None, writer.exception
        async with self.file_budget:
            with await self.aiofiles.open(f'{self.data_dir}/{self.id}/{file_id}', 'rb', buffering=0) as f:
                await self.aiofiles.seek(f, pos)
                assert len(buf) == n
                assert await self.aiofiles.readinto(f, buf) == n

    async def readmany(self, keys):
        out = bytearray(sum(4 + x[2] for x in keys))
        offs = [0] * len(keys)
        s = 0
        for i, n in enumerate(x[2] for x in keys):
            offs[i] = s
            s += n + 4

        async def read_into(key, off):
            _, _, n = key
            struct.pack_into('i', out, off, n)
            off += 4
            await self.read(key, memoryview(out)[off:(off+n)])
        await asyncio.gather(*[read_into(key, off) for key, off in zip(keys, offs)])
        return out

    async def delete(self):
        for i in range(self.current_file):
            await self.aiofiles.remove(f'{self.data_dir}/{self.id}/{i}')
        await self.aiofiles.rmdir(f'{self.data_dir}/{self.id}')


class Sessions:
    def __init__(self, bufsize, data_dir, aiofiles):
        os.mkdir(data_dir)
        # consensus
        self.next_session = 0
        # local
        self.bufsize = bufsize
        self.data_dir = data_dir
        self.aiofiles = aiofiles
        self.sessions = {}

    async def new_session(self):
        id = self.next_session
        self.next_session += 1
        self.sessions[id] = await Session.make(self.bufsize, self.data_dir, id, self.aiofiles)
        return id

    async def delete_session(self, id):
        await self.sessions[id].delete()
        del self.sessions[id]

    async def delete(self):
        for id in self.sessions:
            await self.delete_session(id)
        await self.aiofiles.rmdir(self.data_dir)


class Server:
    def __init__(self, name, binding_host, port, leader, dbuf, aiofiles):
        self.deploy_config = get_deploy_config()
        self.app = web.Application(client_max_size=50 * 1024 * 1024)
        self.routes = web.RouteTableDef()
        self.workers = set()
        self.name = name
        self.binding_host = binding_host
        self.port = port
        self.dbuf = dbuf
        self.leader = leader
        self.leader_url = self.deploy_config.base_url(leader)
        self.aiofiles = aiofiles
        self.shuffle_create_lock = asyncio.Lock()
        self.app.add_routes([
            web.post('/s', self.create),
            web.post('/s/{session}', self.post),
            web.post('/s/{session}/get', self.get),
            web.post('/s/{session}/getmany', self.getmany),
            web.delete('/s/{session}', self.delete),
            web.post('/w', self.register_worker),
            web.get('/w', self.list_workers),
            web.get('/healthcheck', self.healthcheck),
        ])
        self.app.on_cleanup.append(self.cleanup)

    @staticmethod
    def serve(name, bufsize, data_dir, leader, binding_host='0.0.0.0', port=5000):
        aiofiles = af.AIOFiles()
        dbuf = Sessions(bufsize, f'{data_dir}/{port}', aiofiles)
        server = Server(name, binding_host, port, leader, dbuf, aiofiles)
        prefixed_app = server.deploy_config.prefix_application(server.app, server.name)

        async def join_cluster(garbage):
            if server.leader != server.name:
                async def make_request():
                    async with aiohttp.ClientSession(raise_for_status=True,
                                                     timeout=aiohttp.ClientTimeout(total=60)) as cs:
                        async with cs.post(f'{server.leader_url}/w', data=server.name) as resp:
                            assert resp.status == 200
                            await resp.text()
                await utils.retry_all_errors(f'could not join cluster with leader {server.leader} at {server.leader_url}')(make_request)
                log.info(f'joined cluster lead by {server.leader}')

        prefixed_app.on_startup.append(join_cluster)

        web.run_app(prefixed_app,
                    host=server.binding_host,
                    port=server.port,
                    access_log_class=AccessLogger)

    def session(self, request):
        session_id = int(request.match_info['session'])
        session = self.dbuf.sessions.get(session_id)
        if session is None:
            raise web.HTTPNotFound(text=f'{session_id} not found')
        return session

    async def create(self, request):
        session_id = await self.dbuf.new_session()
        async with aiohttp.ClientSession(raise_for_status=True,
                                         timeout=aiohttp.ClientTimeout(total=60)) as cs:
            async def call(worker):
                worker_url = self.deploy_config.base_url(worker)
                async with cs.post(f'{worker_url}/s') as resp:
                    assert resp.status == 200
                    text = resp.text()
                    assert await text == f'{session_id}', f'{text}, {session_id}'
                    log.info(f'successfully created shuffle on {worker} at {worker_url}')
            log.info(f'getting shuffle lock for {session_id}')
            async with self.shuffle_create_lock:
                log.info(f'creating shuffle {session_id} on workers {self.workers}')
                await asyncio.gather(*[call(worker) for worker in self.workers])
        log.info(f'created {session_id}')
        return web.json_response(session_id)

    async def post(self, request):
        session = self.session(request)
        file_id, pos, n = await session.write(await request.read())
        return web.json_response((self.name, file_id, pos, n))

    async def get(self, request):
        session = self.session(request)
        server, file_id, pos, n = await request.json()
        assert server == self.name
        key = file_id, pos, n
        data = bytearray(n)
        await session.read(key, data)
        return web.Response(body=data)

    async def getmany(self, request):
        session = self.session(request)
        keys = await request.json()
        for key in keys:
            if key[0] != self.name:
                raise web.HTTPNotFound(text=f'{self.name} does not have key {key}')
        keys = [key[1:] for key in keys]
        data = await session.readmany(keys)
        return web.Response(body=data)

    async def delete(self, request):
        session_id = int(request.match_info['session'])
        if session_id in self.dbuf.sessions:
            await self.dbuf.delete_session(session_id)
            async with aiohttp.ClientSession(raise_for_status=True,
                                             timeout=aiohttp.ClientTimeout(total=60)) as cs:
                def call(worker):
                    async def f():
                        worker_url = self.deploy_config.base_url(worker)
                        async with cs.delete(f'{worker_url}/s/{session_id}') as resp:
                            assert resp.status == 200
                            await resp.text()
                    return f
                await utils.bounded_gather(*[call(worker) for worker in self.workers])
        return web.Response()

    async def register_worker(self, request):
        worker = await request.text()
        log.info(f'new worker: {worker} + {self.workers}')
        self.workers.add(worker)
        return web.json_response(list(self.workers))

    async def list_workers(self, request):
        return web.json_response([self.name] + list(self.workers))

    async def healthcheck(self, request):
        return web.Response()

    async def cleanup(self, what):
        await self.dbuf.delete()


parser = argparse.ArgumentParser(description='distributed buffer')
parser.add_argument('--name', type=str, help='my name in k8s, e.g. dbuf-0.dbuf', default='localhost')
parser.add_argument('--leader', type=str, help='a deploy_config name that points to a dbuf leader', required=False)
parser.add_argument('--data-dir', type=str, help='directory in which to store data', default='/tmp/shuffler')
parser.add_argument('--host', type=str, help='host to bind to', default='0.0.0.0')
parser.add_argument('--port', type=str, help='port to bind to', default='80')
parser.add_argument('--bufsize', type=int, help='buffer size in MiB', default=512)
args = parser.parse_args()


loop = asyncio.get_event_loop()
bufsize = args.bufsize * 1024 * 1024
try:
    shutil.rmtree(args.data_dir, ignore_errors=True)
    os.mkdir(args.data_dir)
    Server.serve(
        args.name, bufsize, args.data_dir, args.leader, args.host, args.port)
finally:
    shutil.rmtree(args.data_dir, ignore_errors=True)
