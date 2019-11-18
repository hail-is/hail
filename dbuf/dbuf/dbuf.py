import aiohttp as ah
import aiohttp.web as web
import argparse
import asyncio
import concurrent
import multiprocessing as mp
import os
import shutil
import signal
import struct
import sys

from hailtop.config import get_deploy_config

from . import aiofiles as af
from .logging import log
from .retry_forever import retry_forever


def grouped(xs, size):
    n = len(xs)
    i = 0
    while i < n:
        yield xs[i:(i+size)]
        i += size


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
        self.blocking_pool = concurrent.futures.ThreadPoolExecutor()

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
        out = bytearray(sum(x[2] for x in keys) + 4 * len(keys))
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


class DBuf:
    @staticmethod
    async def make(bufsize, aiofiles, data_dir):
        await aiofiles.mkdir(data_dir)
        return DBuf(bufsize, data_dir, aiofiles)

    def __init__(self, bufsize, data_dir, aiofiles):
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
    def __init__(self, hostname, binding_host, port, dbuf, leader_url, aiofiles):
        self.app = web.Application(client_max_size=50 * 1024 * 1024)
        self.routes = web.RouteTableDef()
        self.workers = set()
        self.hostname = hostname
        self.binding_host = binding_host
        self.port = port
        self.root_url = f'http://{self.hostname}:{self.port}'
        self.dbuf = dbuf
        self.leader_url = leader_url
        self.aiofiles = aiofiles
        self.shuffle_create_lock = asyncio.Lock()
        self.app.add_routes([
            web.post('/s', self.create),
            web.post('/s/{session}', self.post),
            web.post('/s/{session}/get', self.get),
            web.post('/s/{session}/getmany', self.getmany),
            web.delete('/s/{session}', self.delete),
            web.post('/w', self.post_worker),
            web.get('/w', self.get_workers),
        ])
        self.app.on_cleanup.append(self.cleanup)

    @staticmethod
    async def serve(hostname, k8s_service, bufsize, data_dir, binding_host='0.0.0.0', port=5000, leader_url=None):
        aiofiles = af.AIOFiles()
        dbuf = await DBuf.make(bufsize, aiofiles, f'{data_dir}/{port}')
        server = Server(hostname, binding_host, port, dbuf, leader_url, aiofiles)
        try:
            app = server.app
            if k8s_service is not None:
                app = get_deploy_config().prefix_application(app, k8s_service)
            runner = web.AppRunner(server.app)
            await runner.setup()
            site = web.TCPSite(runner, host=server.binding_host, port=server.port)
            await site.start()
            log.info(f'server on {server.hostname} bound to {server.binding_host}:{server.port}')
            if server.leader_url is not None:
                async def call():
                    async with ah.ClientSession(raise_for_status=True,
                                                timeout=ah.ClientTimeout(total=60)) as cs:
                        async with cs.post(f'{server.leader_url}/w', data=server.root_url) as resp:
                            assert resp.status == 200
                            await resp.text()
                await retry_forever(
                    call,
                    lambda exc: f'could not join cluster with leader {server.leader_url} due to {exc}')
            while True:
                await asyncio.sleep(1 << 16)
        finally:
            await runner.cleanup()

    def session(self, request):
        session_id = int(request.match_info['session'])
        session = self.dbuf.sessions.get(session_id)
        if session is None:
            raise web.HTTPNotFound(text=f'{session_id} not found')
        return session

    def session_id(self, request):
        session_id = int(request.match_info['session'])
        if session_id not in self.dbuf.sessions:
            raise web.HTTPNotFound(text=f'{session_id} not found')
        return session_id

    async def create(self, request):
        session_id = await self.dbuf.new_session()
        async with ah.ClientSession(raise_for_status=True,
                                    timeout=ah.ClientTimeout(total=60)) as cs:
            async def call(worker):
                async with cs.post(f'{worker}/s') as resp:
                    assert resp.status == 200
                    text = resp.text()
                    assert await text == f'{session_id}', f'{text}, {session_id}'
            async with self.shuffle_create_lock:
                await asyncio.gather(*[call(worker) for worker in self.workers])
        log.info(f'created {session_id}')
        return web.json_response(session_id)

    async def post(self, request):
        session = self.session(request)
        file_id, pos, n = await session.write(await request.read())
        return web.json_response((self.root_url, file_id, pos, n))

    async def get(self, request):
        session = self.session(request)
        server, file_id, pos, n = await request.json()
        assert server == self.root_url
        key = file_id, pos, n
        data = bytearray(n)
        await session.read(key, data)
        return web.Response(body=data)

    async def getmany(self, request):
        session = self.session(request)
        keys = await request.json()
        if len(keys) > 0:
            assert keys[0][0] == self.root_url
        keys = [key[1:] for key in keys]
        data = await session.readmany(keys)
        return web.Response(body=data)

    async def delete(self, request):
        session_id = self.session_id(request)
        await self.dbuf.delete_session(session_id)
        async with ah.ClientSession(raise_for_status=True,
                                    timeout=ah.ClientTimeout(total=60)) as cs:
            async def call(worker):
                async with cs.delete(f'{worker}/s/{session_id}') as resp:
                    assert resp.status == 200
                    await resp.text()
            await asyncio.gather(*[call(worker) for worker in self.workers])
        return web.Response()

    async def post_worker(self, request):
        worker = await request.text()
        log.info(f'new worker: {worker} + {self.workers}')
        self.workers.add(worker)
        return web.json_response(list(self.workers))

    async def get_workers(self, request):
        return web.json_response([self.root_url] + list(self.workers))

    async def cleanup(self, what):
        await self.dbuf.delete()


def server(hostname, k8s_service, bufsize, data_dir, port, i, leader_url):
    loop = asyncio.get_event_loop()

    def die(signum, frame):
        loop.stop()
        sys.exit(signum)
    signal.signal(signal.SIGINT, die)
    signal.signal(signal.SIGTERM, die)
    signal.signal(signal.SIGQUIT, die)
    loop.run_until_complete(Server.serve(hostname, k8s_service, bufsize, data_dir, '0.0.0.0', port, leader_url))


parser = argparse.ArgumentParser(description='distributed buffer')
parser.add_argument('n', type=int, help='number of processes, must be at least one')
parser.add_argument('--hostname', type=str, help='hostname to use to connect to myself', default='localhost')
parser.add_argument('--data-dir', type=str, help='directory in which to store data', default='/tmp/shuffler')
parser.add_argument('--port', type=str, help='port to bind to', default='80')
parser.add_argument('--leader-url', type=str, help='directory in which to store data', required=False)
parser.add_argument('--k8s-service', type=str, help='k8s service name', required=False)
parser.add_argument('--bufsize', type=int, help='buffer size in MiB', default=512)
args = parser.parse_args()

if args.n <= 0:
    print(f'n must be greater than zero, was {args.n}',
          file=sys.stderr)
    sys.exit(1)
try:
    data_dir = args.data_dir
    bufsize = args.bufsize * 1024 * 1024
    port = args.port
    os.mkdir(data_dir)
    servers = []

    def die(signum, frame):
        log.info(f'terminating all servers due to signal {signum}')
        for server in servers:
            try:
                server.terminate()
                server.join()
                server.close()
            except Exception as exc:
                log.error(f'could not shutdown {server} deu to {exc}')
        sys.exit(signum)
    signal.signal(signal.SIGINT, die)
    signal.signal(signal.SIGTERM, die)
    signal.signal(signal.SIGQUIT, die)

    leader_url = args.leader_url
    servers = [mp.Process(target=server, args=(args.hostname, args.k8s_service, bufsize, args.data_dir, port, 0, leader_url))]
    if leader_url is None:
        leader_url = f'http://{args.hostname}:{port}'
    servers.extend(
        mp.Process(target=server, args=(args.hostname, args.k8s_service, bufsize, args.data_dir, port + i, i, leader_url))
        for i in range(1, args.n))
    for server in servers:
        server.start()
    for server in servers:
        server.join()
finally:
    shutil.rmtree(data_dir)
