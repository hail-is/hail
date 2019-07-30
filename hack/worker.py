import os
from shlex import quote as shq
import time
import random
import logging
import asyncio
import aiohttp
from aiohttp import web
import uvloop

uvloop.install()

class UTCFormatter(logging.Formatter):
    converter = time.gmtime

def configure_logging():
    fmt = UTCFormatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s | %(filename)s\t| %(funcName)s:%(lineno)d\t| '
        '%(message)s',
        # FIXME microseconds
        datefmt='%Y-%m-%dT%H:%M:%SZ')

    fh = logging.FileHandler('worker.log')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    root_log = logging.getLogger()
    root_log.addHandler(fh)
    root_log.setLevel(logging.DEBUG)

configure_logging()
log = logging.getLogger('worker')

class ANullContextManager:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass

class WeightedSemaphoreContextManager:
    def __init__(self, sem, weight):
        self.sem = sem
        self.weight = weight

    async def __aenter__(self):
        await self.sem.acquire(self.weight)

    async def __aexit__(self, exc_type, exc, tb):
        await self.sem.release(self.weight)

class WeightedSemaphore:
    def __init__(self, value=1):
        self.value = value
        self.cond = asyncio.Condition()

    async def acquire(self, weight):
        while self.value < weight:
            async with self.cond:
                await self.cond.wait()
        self.value -= weight

    async def release(self, weight):
        self.value += weight
        # FIXME this can be more efficient
        async with self.cond:
            self.cond.notify_all()

    def __call__(self, weight):
        return WeightedSemaphoreContextManager(self, weight)

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

async def docker_delete_container(container_id):
    cmd = f'docker rm {container_id}'
    try:
        log.info(f'running {cmd}')
        await check_shell(cmd)
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
        raise
    except Exception:  # pylint: disable=broad-except
        log.exception(f'{cmd} failed')

async def delete_task_shared(task_token, attempt_token):
    cmd = f'rm -rf /shared/{task_token}-{attempt_token}'
    try:
        log.info(f'running {cmd}')
        await check_shell(cmd)
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
        raise
    except Exception:  # pylint: disable=broad-except
        log.exception(f'{cmd} failed')

async def docker_run(scratch_dir, task_token, task_name, cores, attempt_token, step_name, image, cmd, sem=None):
    full_step = f'task {task_token} {task_name} step {step_name} attempt {attempt_token}'

    if not sem:
        sem = ANullContextManager()

    async with sem:
        container_id = None
        attempts = 0
        while not container_id:
            try:
                docker_cmd = f'docker run -d -v /shared/{task_token}-{attempt_token}:/shared --cpus {cores} --memory {cores * 3.5}g {shq(image)} /bin/bash -c {shq(cmd)}'
                log.info(f'running {full_step}: {docker_cmd}')
                container_id, _ = await check_shell_output(docker_cmd)
                container_id = container_id.decode('utf-8').strip()
            except CalledProcessError as e:
                if attempts < 12 and e.returncode == 125:
                    attempts += 1
                    log.info(f'{full_step} failed, attempt {attempts}, retrying')
                    await asyncio.sleep(5)
                else:
                    log.info(f'{full_step} failed, attempt {attempts}, giving up')
                    raise e

        log.info(f'waiting for {full_step}')

        ec_str, _ = await check_shell_output(f'docker container wait {shq(container_id)}')

    ec_str = ec_str.decode('utf-8').strip()
    ec = int(ec_str)

    log.info(f'{full_step} exit_code {ec}')

    gs_log = f'{scratch_dir}/{task_token}/{attempt_token}/{step_name}.log'
    await check_shell(f'docker logs {container_id} 2>&1 | gsutil cp - {shq(gs_log)}')

    asyncio.ensure_future(docker_delete_container(container_id))

    return ec

class Worker:
    def __init__(self, cores, driver, token):
        self.cores = cores
        self.driver = driver
        self.token = token
        self.free_cores = cores
        self.tasks = set()
        self.last_updated = time.time()
        self.cpu_sem = WeightedSemaphore(cores)

        self.session = None  # set in run()

    async def close(self):
        await self.session.close()
        self.session = None

    async def handle_execute_task(self, request):
        body = await request.json()

        config = body['task']
        # cores = config['cores']
        # if cores > self.free_cores:
        #     return web.HTTPBadRequest(reason='insufficient resources')

        await asyncio.shield(self.handle_execute_task2(config))

        return web.json_response({
            'active_tasks': list(self.tasks)
        })

    async def handle_execute_task2(self, config):
        task_token = config['task_token']
        cores = config['cores']

        self.tasks.add(task_token)
        self.free_cores -= cores

        asyncio.ensure_future(self.run_task(config))

        self.last_updated = time.time()

    async def run_task(self, config):
        task_token = config['task_token']
        try:
            log.info(f'executing task {task_token}')
            await self.run_task2(config)
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception:  # pylint: disable=broad-except
            log.exception(f'caught exception while running task {task_token}')
            # FIMXE notify of internal failure
        finally:
            cores = config['cores']

            self.tasks.remove(task_token)
            self.free_cores += cores

    async def run_task2(self, config):
        scratch_dir = config['scratch_dir']
        task_token = config['task_token']
        task_name = config['task_name']
        cores = config['cores']
        attempt_token = config['attempt_token']
        inputs_cmd = config['inputs_cmd']
        image = config['image']
        cmd = config['command']
        outputs_cmd = config['outputs_cmd']

        log.info(f'running task {task_token} attempt {attempt_token}')

        await check_shell(f'mkdir -p /shared/{task_token}-{attempt_token}')

        input_ec = await docker_run(scratch_dir, task_token, task_name, 1, attempt_token, 'input', 'google/cloud-sdk:255.0.0-alpine', inputs_cmd)

        status = {
            'task_token': task_token,
            'attempt_token': attempt_token,
            'input': input_ec
        }

        if input_ec == 0:
            main_ec = await docker_run(scratch_dir, task_token, task_name, cores, attempt_token, 'main', image, cmd, self.cpu_sem)
            status['main'] = main_ec

            if main_ec == 0:
                output_ec = await docker_run(scratch_dir, task_token, task_name, 1, attempt_token, 'output', 'google/cloud-sdk:255.0.0-alpine', outputs_cmd)
                status['output'] = output_ec

        # cleanup
        asyncio.ensure_future(delete_task_shared(task_token, attempt_token))

        log.info(f'task {task_token} done status {status}')

        async with self.session.post(f'http://{self.driver}:5000/task_complete', json=status):
            log.info(f'task {task_token} status posted')
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

            self.session = aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60))

            await self.register()

            while self.tasks or time.time() - self.last_updated < 60:
                log.info(f'n_tasks {len(self.tasks)} free_cores {self.free_cores} age {time.time() - self.last_updated}')
                await asyncio.sleep(15)

            log.info('idle 60s, exiting')

            body = {'inst_token': self.token}
            async with self.session.post(f'http://{self.driver}:5000/deactivate_worker', json=body):
                log.info('deactivated')
        finally:
            if site:
                await site.stop()
            if app_runner:
                await app_runner.cleanup()

    async def register(self):
        tries = 0
        while True:
            try:
                log.info('registering')
                body = {'inst_token': self.token}
                async with self.session.post(f'http://{self.driver}:5000/activate_worker', json=body) as resp:
                    if resp.status == 200:
                        self.last_updated = time.time()
                        log.info('registered')
                        return
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as e:  # pylint: disable=broad-except
                log.exception('caught exception while registering')
                if tries == 12:
                    log.info('register: giving up')
                    raise e
                tries += 1
            await asyncio.sleep(5 * random.uniform(1, 1.25))


cores = int(os.environ['CORES'])
driver = os.environ['DRIVER']
inst_token = os.environ['INST_TOKEN']
worker = Worker(cores, driver, inst_token)

loop = asyncio.get_event_loop()
loop.run_until_complete(worker.run())
loop.run_until_complete(loop.shutdown_asyncgens())
