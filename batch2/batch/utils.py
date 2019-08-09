import re
import asyncio
from aiohttp import web
import secrets
import logging

log = logging.getLogger('utils')


def abort(code, reason=None):
    if code == 400:
        raise web.HTTPBadRequest(reason=reason)
    if code == 404:
        raise web.HTTPNotFound(reason=reason)
    raise web.HTTPException(reason=reason)


def jsonify(data):
    return web.json_response(data)


def new_token(n=5):
    return ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(n)])


cpu_regex = re.compile(r"^(\d*\.\d+|\d+)([m]?)$")


def parse_cpu(cpu_string):
    match = cpu_regex.fullmatch(cpu_string)
    if match:
        number = float(match.group(1))
        if match.group(2) == 'm':
            number /= 1000
        return number


class CalledProcessError(Exception):
    def __init__(self, command, returncode):
        super().__init__()
        self.command = command
        self.returncode = returncode

    def __str__(self):
        return f'Command {self.command} returned non-zero exit status {self.returncode}.'


async def check_shell(script):
    proc = await asyncio.create_subprocess_exec('/bin/bash', '-c', script)
    await proc.wait()
    if proc.returncode != 0:
        raise CalledProcessError(script, proc.returncode)


async def check_shell_output(script):
    proc = await asyncio.create_subprocess_exec(
        '/bin/bash', '-c', script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    outerr = await proc.communicate()
    if proc.returncode != 0:
        raise CalledProcessError(script, proc.returncode)
    return outerr


class AsyncWorkerPool:
    def __init__(self, parallelism):
        self.queue = asyncio.Queue(maxsize=100)

        for _ in range(parallelism):
            asyncio.ensure_future(self._worker())

    async def _worker(self):
        while True:
            try:
                f, args, kwargs = await self.queue.get()
                await f(*args, **kwargs)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception(f'worker pool caught exception')

    async def call(self, f, *args, **kwargs):
        await self.queue.put((f, args, kwargs))
