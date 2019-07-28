import os
from shlex import quote as shq
import time
import random
import logging
import asyncio
import aiohttp
from aiohttp import web

class UTCFormatter(logging.Formatter):
    converter = time.gmtime

def configure_logging():
    log = logging.getLogger('pipeline')

    fmt = UTCFormatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s | %(filename)s\t| %(funcName)s:%(lineno)d\t| '
        '%(message)s',
        # FIXME microseconds
        datefmt='%Y-%m-%dT%H:%M:%SZ')

    fh = logging.FileHandler('worker.log')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    log.addHandler(fh)

    log.setLevel(logging.DEBUG)
    return log

log = configure_logging()


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


async def docker_run(scratch_dir, attempt_token, name, image, cmd):
    container_id = None
    attempts = 0
    while not container_id:
        try:
            container_id, _ = await check_shell_output(f'docker run -d -v /shared:/shared {shq(image)} /bin/bash -c {shq(cmd)}')
            container_id = container_id.strip()
        except CalledProcessError as e:
            if attempts < 12 and e.returncode == 125:
                attempts += 1
                await asyncio.sleep(5)
            else:
                raise e

    ec_str = await check_shell_output(f'docker container wait {shq(container_id)}')
    ec_str = ec_str.strip()
    ec = int(ec_str)

    log.info(f'ec {name} {ec}')

    gs_log = f'{scratch_dir}/{attempt_token}/{name}.log'
    await check_shell(f'docker logs {container_id} 2>&1 | gsutil cp - {shq(gs_log)}')

    return ec

class Worker:
    def __init__(self, cores, driver, token):
        self.cores = cores
        self.driver = driver
        self.token = token
        self.free_cores = cores
        self.tasks = set()
        self.last_updated = time.time()

    async def handle_execute_task(self, request):
        body = await request.json()

        config = body['task']
        cores = config['cores']
        if cores > self.free_cores:
            return web.HTTPBadRequest(reason='insufficient resources')

        await asyncio.shield(self.handle_execute_task2(config))

        return web.json_response({
            'active_tasks': list(self.tasks)
        })

    async def handle_execute_task2(self, config):
        task_token = config['task_token']
        cores = config['cores']

        self.tasks.add(task_token)
        self.free_cores -= cores

        await asyncio.ensure_future(self.run_task(config))

        self.last_updated = time.time()

    async def run_task(self, config):
        try:
            self.run_task2(config)
        finally:
            task_token = config['task_token']
            cores = config['cores']

            self.tasks.remove(task_token)
            self.free_cores += cores

    async def run_task2(self, config):
        scratch_dir = config['scratch_dir']
        task_token = config['task_token']
        attempt_token = config['attempt_token']
        inputs_cmd = config['inputs_cmd']
        image = config['image']
        cmd = config['command']
        outputs_cmd = config['outputs_cmd']

        input_ec = await docker_run(scratch_dir, attempt_token, 'input', 'google/cloud-sdk:237.0.0-alpine', inputs_cmd)

        status = {
            'task_token': task_token,
            'attempt_token': attempt_token,
            'input': input_ec
        }

        if input_ec == 0:
            main_ec = await docker_run(scratch_dir, attempt_token, 'main', image, cmd)
            status['main'] = main_ec

            if main_ec == 0:
                output_ec = await docker_run(scratch_dir, attempt_token, 'output', 'google/cloud-sdk:237.0.0-alpine', outputs_cmd)
                status['output'] = output_ec

        log.info(f'task {task_token} done: status {status}')

        with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            with session.post('http://{self.driver}:5000/task_complete', json=status) as resp:
                await resp.json()
                self.last_updated = time.time()

    async def run(self):
        app_runner = None
        site = None
        try:
            app = web.Application()
            app.add_routes([
                web.post('/execute_task', self.handle_execute_task)
            ])

            app_runner = web.AppRunner(app)
            await app_runner.setup()
            site = web.TCPSite(app_runner, '0.0.0.0', 5000)
            await site.start()

            await self.run2()
        finally:
            if site:
                await site.stop()
            if app_runner:
                await app_runner.cleanup()

    async def run2(self):
        tries = 0
        while True:
            with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                body = {'inst_token': self.token}
                with session.post('http://{self.driver}:5000/ping', json=body) as resp:
                    if resp.status == 200:
                        self.last_updated = time.time()
                        break
            tries += 1
            if tries == 12:
                # give up
                return
            await asyncio.sleep(5 * random.uniform(1, 1.25))

        while self.tasks or time.time() - self.last_updated < 60:
            await asyncio.sleep(5)


cores = os.environ['CORES']
driver = os.environ['DRIVER']
inst_token = os.environ['INST_TOKEN']
worker = Worker(cores, driver, inst_token)

loop = asyncio.get_event_loop()
loop.run_until_complete(worker.run())
loop.run_until_complete(loop.shutdown_asyncgens())

name = os.environ['NAME']
zone = os.environ['ZONE']

# later
# os.system(f'gcloud -q compute instances delete {name} --zone={zone}')
