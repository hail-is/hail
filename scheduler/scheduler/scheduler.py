import struct
import datetime
import logging
from base64 import b64encode
import asyncio
from aiohttp import web
import jinja2
import aiohttp_jinja2
import uvloop

uvloop.install()


def make_logger():
    fmt = logging.Formatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| '
        '%(funcName)s:%(lineno)d | %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    log = logging.getLogger('batch')
    log.setLevel(logging.INFO)

    logging.basicConfig(handlers=[stream_handler], level=logging.INFO)

    return log


log = make_logger()


async def read_int(reader):
    b = await reader.readexactly(4)
    return struct.unpack('>I', b)[0]


def write_int(writer, i):
    writer.write(struct.pack('>I', i))


async def read_bytes(reader):
    n = await read_int(reader)
    return await reader.readexactly(n)


def write_bytes(writer, b):
    write_int(writer, len(b))
    writer.write(b)


counter = 0


def create_id():
    global counter

    counter = counter + 1
    return counter


executors = set()
clients = set()

jobs = []

pending_jobs = set()

available_executors = set()

task_index = {}


scheduling = False


async def schedule():
    global scheduling

    if scheduling:
        return

    scheduling = True
    while pending_jobs and available_executors:
        e = next(iter(available_executors))
        await e.schedule()

    scheduling = False


# executor messages
# scheduler => executor
EXECUTE = 2

# executor => scheduler
PING = 1
TASKRESULT = 4


class ExecutorConnection:
    def __init__(self, reader, writer, n_cores):
        self.id = create_id()

        log.info(f'client {self.id} connected: {n_cores} cores')

        self.reader = reader
        self.writer = writer
        self.n_cores = n_cores
        self.running = set()
        self.last_message_time = datetime.datetime.now()
        executors.add(self)
        available_executors.add(self)

    async def handle_result(self):
        task_id = await read_int(self.reader)
        res = await read_bytes(self.reader)

        if task_id in task_index:
            t = task_index[task_id]

            duration = datetime.datetime.now() - t.start_time
            log.info(f'client {self.id}: '
                     f'task {t.job.id}/{t.id} complete: {duration}')

            t.set_result(res)

        if len(self.running) == self.n_cores:
            assert self not in available_executors
            available_executors.add(self)
        self.running.remove(t)

        await self.schedule()

    async def handle_ping(self):
        log.info(f'client {self.id}: received ping')

    async def execute(self, t):
        log.info(f'schedule task {t.job.id}/{t.id} on client {self.id}')

        # FIXME time this attempt
        t.start_time = datetime.datetime.now()

        # don't need to lock becuase only called by schedule and
        # schedule is serial
        write_int(self.writer, EXECUTE)
        write_int(self.writer, t.id)
        write_bytes(self.writer, t.f)
        await self.writer.drain()

    async def handler_loop(self):
        try:
            while True:
                cmd = await read_int(self.reader)
                if cmd == PING:
                    await self.handle_ping()
                elif cmd == TASKRESULT:
                    await self.handle_result()
                else:
                    raise ValueError(f'unknown command {cmd}')
                self.last_message_time = datetime.datetime.now()
        except Exception:  # pylint: disable=broad-except
            log.exception(f'client {self.id}: '
                          f'error in handler loop, closing due to exception')
            self.close()

    def close(self):
        executors.remove(self)
        for t in self.running:
            t.job.pending_tasks.append(t)
        self.running = set()

    async def schedule(self):
        while len(self.running) < self.n_cores and pending_jobs:
            j = next(iter(pending_jobs))
            t = j.pending_tasks.pop()
            if not j.pending_tasks:
                pending_jobs.remove(j)

            self.running.add(t)
            if len(self.running) == self.n_cores:
                available_executors.remove(self)

            await self.execute(t)

    def to_dict(self):
        return {
            'id': self.id,
            'n_cores': self.n_cores,
            'n_running': len(self.running)
        }


async def executor_connected_cb(reader, writer):
    log.info('executor connected')
    n_cores = await read_int(reader)
    conn = ExecutorConnection(reader, writer, n_cores)
    asyncio.ensure_future(conn.handler_loop())
    await conn.schedule()


class Job:
    def __init__(self, client, n_tasks):
        self.id = create_id()
        self.client = client

        self.n_tasks = n_tasks
        self.n_submitted = 0
        self.index_task = {}
        self.pending_tasks = []
        self.complete_tasks = set()

        jobs.append(self)

    async def add_task(self, f, index):
        assert index == self.n_submitted
        t = Task(self, f, index)
        self.index_task[index] = t
        self.pending_tasks.append(t)

        self.n_submitted += 1

        if len(self.pending_tasks) == 1:
            pending_jobs.add(self)
            await schedule()

    def ack_task(self, index):
        t = self.index_task.get(index)
        if t is None:
            return
        del self.index_task[index]
        self.complete_tasks.remove(t)
        t.ack()

        if self.is_complete():
            self.client.end_job()

    def is_complete(self):
        return (self.n_submitted == self.n_tasks) and (not self.index_task)

    def to_dict(self):
        n_acknowleged = self.n_submitted - len(self.index_task)
        n_complete = n_acknowleged + len(self.complete_tasks)
        n_running = (self.n_submitted - n_complete) - len(self.pending_tasks)
        return {
            'id': self.id,
            'n_tasks': self.n_tasks,
            'n_submitted': self.n_submitted,
            'n_complete': n_complete,
            'n_running': n_running
        }


class Task:
    def __init__(self, job, f, index):
        self.id = create_id()
        self.job = job
        self.f = f
        self.index = index
        self.start_time = None
        self.result = None

        task_index[self.id] = self

    def set_result(self, result):
        if self.result is not None:
            return
        self.result = result
        self.job.complete_tasks.add(self)

        result_conn = self.job.client.result_conn
        if result_conn:
            asyncio.ensure_future(
                result_conn.task_result(self.index, self.result))

    def ack(self):
        self.result = None
        del task_index[self.id]


# client messages
# client => scheduler
SUBMIT = 5

# scheduler => client
APPTASKRESULT = 6
ACKTASKRESULT = 7


class ClientSubmitConnection:
    def __init__(self, client, reader, writer):
        self.client = client
        self.reader = reader
        self.writer = writer

    async def handle_submit(self):
        n = await read_int(self.reader)

        j = self.client.start_job(n)
        write_int(self.writer, j.n_submitted)
        await self.writer.drain()

        i = j.n_submitted
        while i < n:
            b = await read_bytes(self.reader)
            await j.add_task(b, i)
            i += 1

        # ack
        write_int(self.writer, 0)
        await self.writer.drain()

        log.info(f'job {j.id}, {n} tasks submitted')

        await schedule()

    async def handler_loop(self):
        try:
            while True:
                cmd = await read_int(self.reader)
                if cmd == SUBMIT:
                    await self.handle_submit()
                else:
                    raise ValueError(f'unknown command {cmd}')
        except Exception:  # pylint: disable=broad-except
            log.exception(f'error in handler loop')
        finally:
            self.close()

    def close(self):
        self.writer.close()
        self.client.submit_conn = None
        # new in 3.7
        # await self.writer.wait_closed()


class ClientResultConnection:
    def __init__(self, client, reader, writer):
        self.client = client
        self.reader = reader
        self.writer = writer

    async def handle_ack_task(self):
        index = await read_int(self.reader)
        j = self.client.job
        if j:
            j.ack_task(index)

    async def task_result(self, index, result):
        write_int(self.writer, APPTASKRESULT)
        write_int(self.writer, index)
        write_bytes(self.writer, result)
        await self.writer.drain()

    async def handler_loop(self):
        try:
            while True:
                cmd = await read_int(self.reader)
                if cmd == ACKTASKRESULT:
                    await self.handle_ack_task()
                else:
                    raise ValueError(f'unknown command {cmd}')
        except Exception:  # pylint: disable=broad-except
            log.exception(f'error in handler loop')
        finally:
            self.close()

    def close(self):
        self.writer.close()
        self.client.result_conn = None
        # new in 3.7
        # await self.writer.wait_closed()


token_client = {}


class Client:
    def __init__(self, token):
        self.id = create_id()
        self.token = token
        self.submit_conn = None
        self.result_conn = None

        self.job = None

        clients.add(self)
        log.info(f'client {self.id} created')

    def start_job(self, n):
        if self.job is None:
            self.job = Job(self, n)
        return self.job

    def end_job(self):
        self.job = None

    def set_submit_conn(self, reader, writer):
        if self.submit_conn:
            self.submit_conn.close()
        self.submit_conn = ClientSubmitConnection(self, reader, writer)
        log.info(f'client {self.id} submit connected')
        asyncio.ensure_future(self.submit_conn.handler_loop())

    def set_result_conn(self, reader, writer):
        if self.result_conn:
            self.result_conn.close()
        self.result_conn = ClientResultConnection(self, reader, writer)
        log.info(f'client {self.id} result connected')
        asyncio.ensure_future(self.result_conn.handler_loop())

    def to_dict(self):
        is_disconnected = ((self.submit_conn is None) and
                           (self.result_conn is None))
        return {
            'id': self.id,
            'token': f'{b64encode(self.token)[:4].decode("ascii")}...',
            'is_disconnected': is_disconnected,
            'job_id': self.job.id if self.job else None
        }


async def client_submit_cb(reader, writer):
    token = await read_bytes(reader)

    client = token_client.get(token)
    if client is None:
        client = Client(token)
        token_client[token] = client

    client.set_submit_conn(reader, writer)


async def client_result_cb(reader, writer):
    log.info('here')
    token = await read_bytes(reader)

    client = token_client.get(token)
    if client is None:
        client = Client(token)
        token_client[token] = client

    client.set_result_conn(reader, writer)

app = web.Application()
routes = web.RouteTableDef()


@routes.get('/healthcheck')
async def healthcheck(request):  # pylint: disable=unused-argument
    return web.Response(status=200)


@routes.get('/')
@aiohttp_jinja2.template('index.html')
async def index(request):  # pylint: disable=unused-argument
    return {
        'executors': [e.to_dict() for e in executors],
        'clients': [client.to_dict() for client in clients],
        'jobs': [j.to_dict() for j in reversed(jobs)]
    }


app.add_routes(routes)

aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))


async def on_startup(app):  # pylint: disable=unused-argument
    await asyncio.start_server(executor_connected_cb, host=None, port=5051)
    log.info(f'listening on port {5051} for clients')

    await asyncio.start_server(client_submit_cb, host=None, port=5052)
    log.info(f'listening on port {5052} for clients, submit')

    await asyncio.start_server(client_result_cb, host=None, port=5053)
    log.info(f'listening on port {5053} for clients, result')

app.on_startup.append(on_startup)

web.run_app(app, host='0.0.0.0', port=5000)
