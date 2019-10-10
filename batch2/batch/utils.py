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
        return round(number, 3)
    return None


image_regex = re.compile(r"(?:.+/)?([^:]+)(:(.+))?")


def parse_image_tag(image_string):
    match = image_regex.fullmatch(image_string)
    if match:
        return match.group(3)
    return None


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
