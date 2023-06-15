import argparse
import asyncio
import base64
import json
import os
from shlex import quote as shq
from typing import Dict, List, Optional, Tuple

import kubernetes_asyncio.client
import kubernetes_asyncio.config

from ci.build import BuildConfiguration, Code
from ci.environment import KUBERNETES_SERVER_URL, STORAGE_URI
from ci.github import clone_or_fetch_script
from ci.utils import generate_token
from gear import K8sCache
from hailtop.utils import check_shell_output

BATCH_WORKER_IMAGE = os.environ['BATCH_WORKER_IMAGE']


def populate_secret_host_path(host_path: str, secret_data: Dict[str, bytes]):
    os.makedirs(host_path)
    if secret_data is not None:
        for filename, data in secret_data.items():
            with open(f'{host_path}/{filename}', 'wb') as f:
                f.write(base64.b64decode(data))


class LocalJob:
    def __init__(
        self,
        index: int,
        image: str,
        command: List[str],
        *,
        env: Optional[Dict[str, str]] = None,
        mount_docker_socket: bool = False,
        unconfined: bool = False,
        secrets: Optional[List[Dict[str, str]]] = None,
        service_account: Optional[Dict[str, str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        parents: Optional[List['LocalJob']] = None,
        input_files: Optional[List[Tuple[str, str]]] = None,
        output_files: Optional[List[Tuple[str, str]]] = None,
        **kwargs,
    ):
        self._index = index
        self._image = image
        self._command = command

        self._env = env
        self._mount_docker_socket = mount_docker_socket
        self._unconfined = unconfined
        self._parents = parents
        self._secrets = secrets
        self._service_account = service_account
        self._attributes = attributes
        self._input_files = input_files
        self._output_files = output_files
        self._kwargs = kwargs

        self._succeeded: Optional[bool] = None


async def docker_run(*args: str):
    script = ' '.join([shq(a) for a in args])
    outerr = await check_shell_output(script)
    print(f'Container output: {outerr[0]!r}\n' f'Container error: {outerr[1]!r}')

    cid = outerr[0].decode('ascii').strip()

    outerr = await check_shell_output(f'docker wait {cid}')
    exit_code = int(outerr[0].decode('ascii').strip())
    return cid, exit_code == 0


class LocalBatchBuilder:
    def __init__(self, attributes: Dict[str, str], callback: Optional[str]):
        self._attributes = attributes
        self._callback = callback
        self._jobs: List[LocalJob] = []

    @property
    def attributes(self) -> Dict[str, str]:
        return self._attributes

    @property
    def callback(self) -> Optional[str]:
        return self._callback

    def create_job(self, image: str, command: List[str], **kwargs):
        index = len(self._jobs)
        job = LocalJob(index, image, command, **kwargs)
        self._jobs.append(job)
        return job

    async def run(self):
        cwd = os.getcwd()
        assert cwd.startswith('/')

        batch_token = self._attributes['token']
        root = f'{cwd}/_/{batch_token}'

        os.makedirs(f'{root}/shared')

        prefix = f'{STORAGE_URI}/build/{batch_token}'

        for j in self._jobs:
            assert j._attributes
            job_name = j._attributes.get('name')

            print(f'{j._index}: {job_name}: running...')

            if j._parents:
                for p in j._parents:
                    assert p._succeeded is not None
                    if not p._succeeded:
                        print(f'{j._index}: {job_name}: SKIPPED: parent {p._index} failed')
                        j._succeeded = False

            if j._succeeded is False:
                continue

            job_root = f'{root}/{j._index}'

            os.makedirs(f'{job_root}/io')
            os.makedirs(f'{job_root}/secrets')

            if j._input_files:
                files = []
                for src, dest in j._input_files:
                    assert src.startswith(prefix), (prefix, src)
                    files.append(
                        {
                            'from': f'/shared{src[len(prefix):]}',
                            'to': dest,
                        }
                    )
                input_cid, input_ok = await docker_run(
                    'docker',
                    'run',
                    '-d',
                    '-v',
                    f'{root}/shared:/shared',
                    '-v',
                    f'{job_root}/io:/io',
                    '--entrypoint',
                    '/usr/bin/python3',
                    BATCH_WORKER_IMAGE,
                    '-m',
                    'hailtop.aiotools.copy',
                    json.dumps(None),
                    json.dumps(files),
                )

                print(f'{j._index}: {job_name}/input: {input_cid} {"OK" if input_ok else "FAILED"}')
            else:
                input_ok = True

            if input_ok:
                mount_options = ['-v', f'{job_root}/io:/io']

                env_options = []
                if j._env:
                    for key, value in j._env.items():
                        env_options.extend(['-e', f'{key}={value}'])

                # Reboot the cache on each use.  The kube client isn't
                # refreshing tokens correctly.
                # https://github.com/kubernetes-client/python/issues/741
                # Note, that is in the kubenetes-client repo, the
                # kubernetes_asyncio.  I'm assuming it has the same
                # issue.
                k8s_client = kubernetes_asyncio.client.CoreV1Api()
                try:
                    k8s_cache = K8sCache(k8s_client)

                    if j._service_account:
                        namespace = j._service_account['namespace']
                        name = j._service_account['name']

                        secret = await k8s_cache.read_secret(f'{name}-token', namespace)

                        token = base64.b64decode(secret.data['token']).decode()
                        cert = secret.data['ca.crt']

                        kube_config = f'''
apiVersion: v1
clusters:
- cluster:
    certificate-authority: /.kube/ca.crt
    server: {KUBERNETES_SERVER_URL}
  name: default-cluster
contexts:
- context:
    cluster: default-cluster
    user: {namespace}-{name}
    namespace: {namespace}
  name: default-context
current-context: default-context
kind: Config
preferences: {{}}
users:
- name: {namespace}-{name}
  user:
    token: {token}
'''

                        dot_kube_dir = f'{job_root}/secrets/.kube'

                        os.makedirs(dot_kube_dir)
                        with open(f'{dot_kube_dir}/config', 'w', encoding='utf-8') as f:
                            f.write(kube_config)
                        with open(f'{dot_kube_dir}/ca.crt', 'w', encoding='utf-8') as f:
                            f.write(base64.b64decode(cert).decode())
                        mount_options.extend(['-v', f'{dot_kube_dir}:/.kube'])
                        env_options.extend(['-e', 'KUBECONFIG=/.kube/config'])

                    secrets = j._secrets
                    if secrets:
                        k8s_secrets = await asyncio.gather(
                            *[k8s_cache.read_secret(secret['name'], secret['namespace']) for secret in secrets]
                        )

                        for secret, k8s_secret in zip(secrets, k8s_secrets):
                            secret_host_path = f'{job_root}/secrets/{k8s_secret.metadata.name}'

                            populate_secret_host_path(secret_host_path, k8s_secret.data)

                            mount_options.extend(['-v', f'{secret_host_path}:{secret["mount_path"]}'])

                    if j._mount_docker_socket:
                        mount_options.extend(['-v', '/var/run/docker.sock:/var/run/docker.sock'])

                    if j._unconfined:
                        security_options = [
                            '--security-opt',
                            'seccomp=unconfined',
                            '--security-opt',
                            'apparmor=unconfined',
                        ]
                    else:
                        security_options = []

                    main_cid, main_ok = await docker_run(
                        'docker',
                        'run',
                        '-d',
                        *env_options,
                        *mount_options,
                        *security_options,
                        '--entrypoint',
                        j._command[0],
                        j._image,
                        *j._command[1:],
                    )
                    print(f'{j._index}: {job_name}/main: {main_cid} {"OK" if main_ok else "FAILED"}')
                finally:
                    await k8s_client.api_client.rest_client.pool_manager.close()
            else:
                main_ok = False
                print(f'{j._index}: {job_name}/main: SKIPPED: input failed')

            if j._output_files:
                if main_ok:
                    files = []
                    for src, dest in j._output_files:
                        assert dest.startswith(prefix), (prefix, dest)
                        files.append(
                            {
                                'from': src,
                                'to': f'/shared{dest[len(prefix):]}',
                            }
                        )
                    output_cid, output_ok = await docker_run(
                        'docker',
                        'run',
                        '-d',
                        '-v',
                        f'{root}/shared:/shared',
                        '-v',
                        f'{job_root}/io:/io',
                        '--entrypoint',
                        '/usr/bin/python3',
                        BATCH_WORKER_IMAGE,
                        '-m',
                        'hailtop.aiotools.copy',
                        json.dumps(None),
                        json.dumps(files),
                    )
                    print(f'{j._index}: {job_name}/output: {output_cid} {"OK" if output_ok else "FAILED"}')
                else:
                    output_ok = False
                    print(f'{j._index}: {job_name}/output: SKIPPED: main failed')
            else:
                output_ok = True

            j._succeeded = input_ok and main_ok and output_ok


class Branch(Code):
    def __init__(self, owner: str, repo: str, branch: str, sha: str, extra_config: Dict[str, str]):
        self._owner = owner
        self._repo = repo
        self._branch = branch
        self._sha = sha
        self._extra_config = extra_config

    def short_str(self) -> str:
        return f'br-{self._owner}-{self._repo}-{self._branch}'

    def repo_url(self) -> str:
        return f'https://github.com/{self._owner}/{self._repo}'

    def config(self) -> Dict[str, str]:
        config = {
            'checkout_script': self.checkout_script(),
            'branch': self._branch,
            'repo': f'{self._owner}/{self._repo}',
            'repo_url': self.repo_url(),
            'sha': self._sha,
        }
        config.update(self._extra_config)
        return config

    def checkout_script(self) -> str:
        return f'''
{clone_or_fetch_script(self.repo_url())}

git checkout {shq(self._sha)}
'''

    def repo_dir(self) -> str:
        return '.'


async def main():
    await kubernetes_asyncio.config.load_kube_config()

    parser = argparse.ArgumentParser(description='Bootstrap a Hail as a service installation.')

    parser.add_argument(
        '--extra-code-config', dest='extra_code_config', default='{}', help='Extra code config in JSON format.'
    )
    parser.add_argument(
        'branch', help='Github branch to run.  It should be the same branch bootstrap.py is being run from.'
    )
    parser.add_argument('sha', help='SHA of the git commit to run.  It should match the branch.')
    parser.add_argument('steps', help='The requested steps to execute.')

    args = parser.parse_args()

    branch_pieces = args.branch.split(":")
    assert len(branch_pieces) == 2, f'{branch_pieces} {args.branch}'

    repo_pieces = branch_pieces[0].split("/")
    assert len(repo_pieces) == 2, f'{repo_pieces} {branch_pieces[0]}'
    owner = repo_pieces[0]
    repo_name = repo_pieces[1]

    branch_name = branch_pieces[1]

    extra_code_config = json.loads(args.extra_code_config)

    scope = 'deploy'
    code = Branch(owner, repo_name, branch_name, args.sha, extra_code_config)

    steps = [s.strip() for s in args.steps.split(',')]

    with open('build.yaml', 'r', encoding='utf-8') as f:
        config = BuildConfiguration(code, f.read(), scope, requested_step_names=steps)

    token = generate_token()
    batch = LocalBatchBuilder(attributes={'token': token}, callback=None)
    config.build(batch, code, scope)

    await batch.run()


asyncio.get_event_loop().run_until_complete(main())
