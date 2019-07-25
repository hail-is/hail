import abc
import os
import subprocess as sp
import re
import uuid
import secrets
import json
import asyncio
from aiohttp import web
from shlex import quote as shq
from hailtop.batch_client.client import BatchClient, Job
import aiohttp

from .resource import InputResourceFile, TaskResourceFile
from .utils import PipelineException

class Backend:
    @abc.abstractmethod
    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        return


class LocalBackend(Backend):
    """
    Backend that executes pipelines on a local computer.

    Examples
    --------

    >>> local_backend = LocalBackend(tmp_dir='/tmp/user/')
    >>> p = Pipeline(backend=local_backend)

    Parameters
    ----------
    tmp_dir: :obj:`str`, optional
        Temporary directory to use.
    gsa_key_file :obj:`str`, optional
        Mount a file with a gsa key to `/gsa-key/privateKeyData`. Only used if a
        task specifies a docker image. This option will override the value set by
        the environment variable `HAIL_PIPELINE_GSA_KEY_FILE`.
    extra_docker_run_flags :obj:`str`, optional
        Additional flags to pass to `docker run`. Only used if a task specifies
        a docker image. This option will override the value set by the environment
        variable `HAIL_PIPELINE_EXTRA_DOCKER_RUN_FLAGS`.
    """

    def __init__(self, tmp_dir='/tmp/', gsa_key_file=None, extra_docker_run_flags=None):
        self._tmp_dir = tmp_dir

        flags = ''

        if extra_docker_run_flags is not None:
            flags += extra_docker_run_flags
        elif os.environ.get('HAIL_PIPELINE_EXTRA_DOCKER_RUN_FLAGS') is not None:
            flags += os.environ['HAIL_PIPELINE_EXTRA_DOCKER_RUN_FLAGS']

        if gsa_key_file is None:
            gsa_key_file = os.environ.get('HAIL_PIPELINE_GSA_KEY_FILE')
        if gsa_key_file is not None:
            flags += f' -v {gsa_key_file}:/gsa-key/privateKeyData'

        self._extra_docker_run_flags = flags

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable=R0915
        tmpdir = self._get_scratch_dir()

        script = ['#!/bin/bash',
                  'set -e' + 'x' if verbose else '',
                  '\n',
                  '# change cd to tmp directory',
                  f"cd {tmpdir}",
                  '\n']

        copied_input_resource_files = set()
        os.makedirs(tmpdir + 'inputs/', exist_ok=True)

        def copy_input(task, r):
            if isinstance(r, InputResourceFile):
                if r not in copied_input_resource_files:
                    copied_input_resource_files.add(r)

                    if r._input_path.startswith('gs://'):
                        return [f'gsutil cp {r._input_path} {r._get_path(tmpdir)}']
                    else:
                        absolute_input_path = shq(os.path.realpath(r._input_path))
                        if task._image is not None:  # pylint: disable-msg=W0640
                            return [f'cp {absolute_input_path} {r._get_path(tmpdir)}']
                        else:
                            return [f'ln -sf {absolute_input_path} {r._get_path(tmpdir)}']
                else:
                    return []
            else:
                assert isinstance(r, TaskResourceFile)
                return []

        def copy_external_output(r):
            def cp(dest):
                if not dest.startswith('gs://'):
                    dest = os.path.abspath(dest)
                    directory = os.path.dirname(dest)
                    os.makedirs(directory, exist_ok=True)
                    return 'cp'
                else:
                    return 'gsutil cp'

            if isinstance(r, InputResourceFile):
                return [f'{cp(dest)} {shq(r._input_path)} {shq(dest)}'
                        for dest in r._output_paths]
            else:
                assert isinstance(r, TaskResourceFile)
                return [f'{cp(dest)} {r._get_path(tmpdir)} {shq(dest)}'
                        for dest in r._output_paths]

        write_inputs = [x for r in pipeline._input_resources for x in copy_external_output(r)]
        if write_inputs:
            script += ["# Write input resources to output destinations"]
            script += write_inputs
            script += ['\n']

        for task in pipeline._tasks:
            os.makedirs(tmpdir + task._uid + '/', exist_ok=True)

            script.append(f"# {task._uid} {task.name if task.name else ''}")

            script += [x for r in task._inputs for x in copy_input(task, r)]

            resource_defs = [r._declare(tmpdir) for r in task._mentioned]

            if task._image:
                defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
                cmd = " && ".join(task._command)
                memory = f'-m {task._memory}' if task._memory else ''
                cpu = f'--cpus={task._cpu}' if task._cpu else ''

                script += [f"docker run "
                           f"{self._extra_docker_run_flags} "
                           f"-v {tmpdir}:{tmpdir} "
                           f"-w {tmpdir} "
                           f"{memory} "
                           f"{cpu} "
                           f"{task._image} /bin/bash "
                           f"-c {shq(defs + cmd)}",
                           '\n']
            else:
                script += resource_defs
                script += task._command

            script += [x for r in task._external_outputs for x in copy_external_output(r)]
            script += ['\n']

        script = "\n".join(script)
        if dry_run:
            print(script)
        else:
            try:
                sp.check_output(script, shell=True)
            except sp.CalledProcessError as e:
                print(e)
                print(e.output)
                raise
            finally:
                if delete_scratch_on_exit:
                    sp.run(f'rm -rf {tmpdir}', shell=True)

        print('Pipeline completed successfully!')

    def _get_scratch_dir(self):
        def _get_random_name():
            directory = self._tmp_dir + '/pipeline-{}/'.format(uuid.uuid4().hex[:12])

            if os.path.isdir(directory):
                return _get_random_name()
            else:
                os.makedirs(directory, exist_ok=True)
                return directory

        return _get_random_name()


class BatchBackend(Backend):
    """
    Backend that executes pipelines on a Kubernetes cluster using `batch`.

    Examples
    --------

    >>> batch_backend = BatchBackend(tmp_dir='http://localhost:5000')
    >>> p = Pipeline(backend=batch_backend)

    Parameters
    ----------
    url: :obj:`str`
        URL to batch server.
    """

    def __init__(self, url):
        session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
        self._batch_client = BatchClient(session, url)

    def close(self):
        self._batch_client.close()

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable-msg=R0915
        bucket = self._batch_client.bucket
        subdir_name = 'pipeline-{}'.format(uuid.uuid4().hex[:12])

        remote_tmpdir = f'gs://{bucket}/pipeline/{subdir_name}'
        local_tmpdir = f'/io/pipeline/{subdir_name}'

        default_image = 'ubuntu'

        attributes = pipeline.attributes
        if pipeline.name is not None:
            attributes['name'] = pipeline.name

        batch = self._batch_client.create_batch(attributes=attributes)

        n_jobs_submitted = 0
        used_remote_tmpdir = False

        task_to_job_mapping = {}
        jobs_to_command = {}
        commands = []

        bash_flags = 'set -e' + ('x' if verbose else '') + '; '

        activate_service_account = 'gcloud -q auth activate-service-account ' \
                                   '--key-file=/gsa-key/privateKeyData'

        def copy_input(r):
            if isinstance(r, InputResourceFile):
                return [(r._input_path, r._get_path(local_tmpdir))]
            else:
                assert isinstance(r, TaskResourceFile)
                return [(r._get_path(remote_tmpdir), r._get_path(local_tmpdir))]

        def copy_internal_output(r):
            assert isinstance(r, TaskResourceFile)
            return [(r._get_path(local_tmpdir), r._get_path(remote_tmpdir))]

        def copy_external_output(r):
            if isinstance(r, InputResourceFile):
                return [(r._input_path, dest) for dest in r._output_paths]
            else:
                assert isinstance(r, TaskResourceFile)
                return [(r._get_path(local_tmpdir), dest) for dest in r._output_paths]

        write_external_inputs = [x for r in pipeline._input_resources for x in copy_external_output(r)]
        if write_external_inputs:
            def _cp(src, dst):
                return f'gsutil -m cp -R {src} {dst}'

            write_cmd = bash_flags + activate_service_account + ' && ' + \
                        ' && '.join([_cp(*files) for files in write_external_inputs])

            if dry_run:
                commands.append(write_cmd)
            else:
                j = batch.create_job(image='google/cloud-sdk:237.0.0-alpine',
                                     command=['/bin/bash', '-c', write_cmd],
                                     attributes={'name': 'write_external_inputs'})
                jobs_to_command[j] = write_cmd
                n_jobs_submitted += 1

        for task in pipeline._tasks:
            inputs = [x for r in task._inputs for x in copy_input(r)]

            outputs = [x for r in task._internal_outputs for x in copy_internal_output(r)]
            if outputs:
                used_remote_tmpdir = True
            outputs += [x for r in task._external_outputs for x in copy_external_output(r)]

            resource_defs = [r._declare(directory=local_tmpdir) for r in task._mentioned]

            if task._image is None:
                if verbose:
                    print(f"Using image '{default_image}' since no image was specified.")

            make_local_tmpdir = f'mkdir -p {local_tmpdir}/{task._uid}/; '
            defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
            task_command = [cmd.strip() for cmd in task._command]

            cmd = bash_flags + make_local_tmpdir + defs + " && ".join(task_command)
            if dry_run:
                commands.append(cmd)
                continue

            parents = [task_to_job_mapping[t] for t in task._dependencies]

            attributes = {'task_uid': task._uid}
            if task.name:
                attributes['name'] = task.name
            attributes.update(task.attributes)

            resources = {'requests': {}}
            if task._cpu:
                resources['requests']['cpu'] = task._cpu
            if task._memory:
                resources['requests']['memory'] = task._memory

            j = batch.create_job(image=task._image if task._image else default_image,
                                 command=['/bin/bash', '-c', cmd],
                                 parents=parents,
                                 attributes=attributes,
                                 resources=resources,
                                 input_files=inputs if len(inputs) > 0 else None,
                                 output_files=outputs if len(outputs) > 0 else None,
                                 pvc_size=task._storage)
            n_jobs_submitted += 1

            task_to_job_mapping[task] = j
            jobs_to_command[j] = cmd

        if dry_run:
            print("\n\n".join(commands))
            return

        if delete_scratch_on_exit and used_remote_tmpdir:
            parents = list(jobs_to_command.keys())
            rm_cmd = f'gsutil rm -r {remote_tmpdir}'
            cmd = bash_flags + f'{activate_service_account} && {rm_cmd}'
            j = batch.create_job(
                image='google/cloud-sdk:237.0.0-alpine',
                command=['/bin/bash', '-c', cmd],
                parents=parents,
                attributes={'name': 'remove_tmpdir'},
                always_run=True)
            jobs_to_command[j] = cmd
            n_jobs_submitted += 1

        batch = batch.submit()

        jobs_to_command = {j.id: cmd for j, cmd in jobs_to_command.items()}

        if verbose:
            print(f'Submitted batch {batch.id} with {n_jobs_submitted} jobs:')
            for jid, cmd in jobs_to_command.items():
                print(f'{jid}: {cmd}')

        status = batch.wait()

        if status['state'] == 'success':
            print('Pipeline completed successfully!')
            return

        failed_jobs = [((j['batch_id'], j['job_id']), j['exit_code']) for j in status['jobs'] if 'exit_code' in j and any([ec != 0 for _, ec in j['exit_code'].items()])]

        fail_msg = ''
        for jid, ec in failed_jobs:
            ec = Job.exit_code(ec)
            job = self._batch_client.get_job(*jid)
            log = job.log()
            name = job.status()['attributes'].get('name', None)
            fail_msg += (
                f"Job {jid} failed with exit code {ec}:\n"
                f"  Task name:\t{name}\n"
                f"  Command:\t{jobs_to_command[jid]}\n"
                f"  Log:\t{log}\n")

        raise PipelineException(fail_msg)


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


ZONE = os.environ['ZONE']


class HackRunner:
    def __init__(self, pipeline, verbose):
        self.pipeline = pipeline
        self.verbose = verbose

        self.semaphore = asyncio.Semaphore(10)

        self.ready = asyncio.Queue()

        self.n_complete = 0

        self.task_n_deps = {}
        for t in pipeline._tasks:
            n = len(t._dependencies)
            self.task_n_deps[t] = n

        self.task_children = {t: set() for t in pipeline._tasks}
        for t in pipeline._tasks:
            for d in t._dependencies:
                self.task_children[d].add(t)

        self.task_token = {}
        # token of complete attempt
        self.token_task = {}

        self.task_state = {}

        self.app = web.Application()
        self.app.add_routes([
            web.post('/status', self.handle_status),
            web.post('/shutdown', self.handle_shutdown)
        ])

    async def handle_status(self, request):
        status = await request.json()
        print(f'POST /status status {json.dumps(status)}')
        await self.mark_complete(status)
        return web.Response()

    async def handle_shutdown(self, request):
        body = await request.json()
        print(f'POST /shutdown body {json.dumps(body)}')
        await self.shutdown(body['token'])
        return web.Response()

    def gs_input_path(self, r):
        if isinstance(r, InputResourceFile):
            return r._input_path
        else:
            assert isinstance(r, TaskResourceFile)
            assert r._source
            token = self.task_token[r._source]
            return r._get_path(f'gs://hail-cseed/cs-hack/tmp/{token}')

    def gs_output_paths(self, r, token):
        assert isinstance(r, TaskResourceFile)
        output_paths = [r._get_path(f'gs://hail-cseed/cs-hack/tmp/{token}')]
        if r._output_paths:
            for p in r._output_paths:
                output_paths.append(p)
        return output_paths

    async def shutdown(self, token):
        self.semaphore.release()

        t = self.token_task[token]

        print(f'INFO: shutdown task {t._uid} {t.name} token {token}')

        if t in self.task_state:
            return

        print(f'INFO: rescheduling task {t._uid} {t.name}')
        await self.ready.put(t)

    async def notify_children(self, t):
        for c in self.task_children[t]:
            n = self.task_n_deps[c]
            assert n > 0
            n -= 1
            self.task_n_deps[c] = n
            if n == 0:
                await self.ready.put(c)

    async def set_state(self, t, state, token):
        if t in self.task_state:
            return

        print(f'INFO: set_state task {t._uid} {t.name} state {state} token {token}')
        
        self.task_state[t] = state
        if token:
            self.task_token[t] = token
        self.n_complete += 1
        await self.notify_children(t)

        if self.n_complete == len(self.pipeline._tasks):
            await self.ready.put(None)

    async def mark_complete(self, status):
        token = status['token']
        t = self.token_task[token]

        state = 'success'
        for name in ['input', 'main', 'output']:
            ec = status.get(name)
            if ec is not None and ec != 0:
                state = 'failure'

        await self.set_state(t, state, token)

    async def launch(self, t):
        assert t._image

        if t in self.task_state:
            return

        if any(self.task_state[p] != 'success' for p in t._dependencies):
            await self.set_state(t, 'cancelled', None)
            return

        token = secrets.token_hex(16)
        self.token_task[token] = t

        print(f'INFO: launching task {t._uid} {t.name} token {token}')

        inputs_cmd = ' && '.join([
            f'gsutil -m cp -r {shq(self.gs_input_path(i))} {shq(i._get_path("/shared"))}'
            for i in t._inputs
        ]) if t._inputs else None

        bash_flags = 'set -e' + ('x' if self.verbose else '') + '; '
        defs = ''.join([r._declare('/shared') + '; ' for r in t._mentioned])
        make_local_tmpdir = f'mkdir -p /shared/{t._uid}'
        cmd = bash_flags + defs + make_local_tmpdir + ' && (' + ' && '.join(t._command) + ')'

        outputs = t._internal_outputs.union(t._external_outputs)
        outputs_cmd = ' && '.join([
            f'gsutil -m cp -r {shq(o._get_path("/shared"))} {shq(output_path)}'
            for o in outputs for output_path in self.gs_output_paths(o, token)
        ]) if t._outputs else None

        config = {
            'master': 'cs-hack-master',
            'uid': t._uid,
            'name': t.name,
            'token': token,
            'inputs_cmd': inputs_cmd,
            'image': t._image,
            'command': cmd,
            'outputs_cmd': outputs_cmd
        }

        await check_shell(f'echo {shq(json.dumps(config))} | gsutil cp - gs://hail-cseed/cs-hack/tmp/{token}/config.json')

        if t._cpu:
            cpu = round_up(t._cpu)
        else:
            cpu = 1

        # max out at 16
        cores = 1
        while cores < 16 and cores < req_cpu:
            cores *= 2

        if t._memory:
            memory = float(t._memory)
        else:
            memory = cores * 3.75

        memory_per_core = memory / cores
        if cores == 1:
            machine_type == 'standard'
        elif memory_per_core < 1.2:
            machine_type = 'highcpu'
        elif memory_per_core < 4:
            machine_type = 'standard'
        else:
            machine_type = 'highmem'

        if t._storage:
            storage = t._storage

            pat = '(?P<value>[0-9\\.]+)(?P<unit>[KMGTP]i?B?)?'
            m = re.match(pat, storage)

            if not m:
                raise ValueError(f'could not convert to size: {storage}')
            value = float(m['value'])

            unit = m['unit']
            if not unit:
                multiplier = 1
            else:
                end = -1
                if unit[end] == 'B':
                    end -= 1
                if unit[end] == 'i':
                    base = 1024
                else:
                    base = 1000
                exponents = {'K': 1, 'M': 2, 'G': 3, 'T': 4}
                e = exponents[unit[0]]
                multiplier = base**e
            storage_gb = round_up((value*multiplier)/(1000**3))
            # FIXME consider max 200GB
            if storage_gb < 20:
                storage_gb = 20
        else:
            storage_gb = 20

        print(f'INFO: requested cpu {t._cpu} mem {t._memory} disk {t._storage}, allocated n1-{machine_type}-{cores}, {storage_gb}GB')

        await check_shell(f'gcloud -q compute instances create cs-hack-{token} --zone={ZONE} --async --machine-type=n1-{machine_type}-{cores} --network=default --network-tier=PREMIUM --metadata=master=cs-hack-master,token={token},startup-script-url=gs://hail-cseed/cs-hack/task-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --scopes=https://www.googleapis.com/auth/cloud-platform --image=cs-hack --boot-disk-size={storage_gb}GB --boot-disk-type=pd-ssd')

    async def run(self):
        print(f'INFO: running pipeline')

        app_runner = web.AppRunner(self.app)
        await app_runner.setup()
        site = web.TCPSite(app_runner, '0.0.0.0', 5000)
        await site.start()

        for t, n in self.task_n_deps.items():
            if n == 0:
                await self.ready.put(t)

        while True:
            t = await self.ready.get()
            if not t:
                return
            await self.semaphore.acquire()
            await self.launch(t)

        await app_runner.cleanup()

        print(f'INFO: pipeline finished')

def round_up(x):
    i = int(x)
    if x > i:
        i += 1

class HackBackend(Backend):
    def __init__(self):
        pass

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        runner = HackRunner(pipeline, verbose)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
