import os
from shlex import quote as shq
import base64
import asyncio

import kubernetes_asyncio as kube

from hailtop.utils import check_shell_output

from ci.build import BuildConfiguration, Code
from ci.github import clone_or_fetch_script
from ci.utils import generate_token

from batch.driver.k8s_cache import K8sCache

KUBERNETES_SERVER_URL = 'https://34.71.246.49'


def populate_secret_host_path(host_path, secret_data):
    os.makedirs(host_path)
    if secret_data is not None:
        for filename, data in secret_data.items():
            with open(f'{host_path}/{filename}', 'wb') as f:
                f.write(base64.b64decode(data))


class LocalJob:
    def __init__(self, index, image, command, *,
                 env=None, mount_docker_socket=False, secrets=None, service_account=None, parents=None,
                 input_files=None, output_files=None,
                 **kwargs):
        self._index = index
        self._image = image
        self._command = command

        self._env = env
        self._mount_docker_socket = mount_docker_socket
        self._parents = parents
        self._secrets = secrets
        self._service_account = service_account
        self._input_files = input_files
        self._output_files = output_files
        self._kwargs = kwargs

        self._done = False
        self._succeeded = None


async def docker_run(*args):
    script = ' '.join([shq(a) for a in args])
    outerr = await check_shell_output(script)
    
    cid = outerr[0].decode('ascii').strip()
    
    outerr = await check_shell_output(f'docker wait {cid}')
    
    exit_code = int(outerr[0].decode('ascii').strip())
    return cid, exit_code == 0


class LocalBatchBuilder:
    def __init__(self, attributes, callback):
        self._attributes = attributes
        self._callback = callback
        self._jobs = []

    @property
    def attributes(self):
        return self._attributes

    @property
    def callback(self):
        return self._callback

    def create_job(self, image, command, **kwargs):
        index = len(self._jobs)
        job = LocalJob(index, image, command, **kwargs)
        self._jobs.append(job)
        return job

    async def run(self, token):
        cwd = os.getcwd()
        assert cwd.startswith('/')
        
        token = generate_token()
        root = f'{cwd}/_/{token}'
        
        await kube.config.load_kube_config()
        k8s_client = kube.client.CoreV1Api()
        k8s_cache = K8sCache(k8s_client, refresh_time=5)

        os.makedirs(f'{root}/shared')

        # FIXME hack
        prefix = 'gs://dummy/build/{token}'
        
        for j in self._jobs:
            if j._parents:
                for p in j._parents:
                    assert p._done
                    if not p._succeeded:
                        print(f'{j._index}: skipping: parent {p._index} failed')
                        j._done = True
                        j._failed = True

            job_root = f'{root}/{j._index}'

            os.makedirs(f'{job_root}/io')
            os.makedirs(f'{job_root}/secrets')

            if j._input_files:
                copy_script = 'set -ex\n'
                for src, dest in j._input_files:
                    assert src.startswith(prefix)
                    src = f'/shared{src[len(prefix):]}'
                    copy_script = copy_script + 'cp -a {src} {dest}\n'
                input_cid, input_ok = await docker_run(
                    'docker', 'run', '-d', 'ubuntu:18.04', '-v', f'{root}/shared:/shared', '-v', f'{job_root}/io:/io', copy_script)

                print(f'{j._index}/input: {input_cid} {input_ok}')
            else:
                input_ok = True

            if input_ok:
                mount_options = [
                    '-v', f'{root}/io:/io'
                ]
                
                env_options = []
                if j._env:
                    for key, value in j._env:
                        env_options.extend([
                            '-e', f'{key}={value}'])
                if j._service_account:
                    namespace = j._service_account['namespace']
                    name = j._service_account['name']

                    sa = await k8s_cache.read_service_account(name, namespace, 5)
                    assert len(sa.secrets) == 1

                    token_secret_name = sa.secrets[0].name

                    secret = await k8s_cache.read_secret(token_secret_name, namespace, 5)

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
                    with open(f'{dot_kube_dir}/config', 'w') as f:
                        f.write(kube_config)
                    with open(f'{dot_kube_dir}/ca.crt', 'w') as f:
                        f.write(base64.b64decode(cert).decode())
                    mount_options.extend([
                        '-v', f'{dot_kube_dir}:/.kube'
                    ])
                    env_options.extend([
                        '-e', 'KUBECONFIG=/.kube/config'])

                secrets = j._secrets
                if secrets:
                    k8s_secrets = await asyncio.gather(*[
                        k8s_cache.read_secret(
                            secret['name'], secret['namespace'],
                            5)
                        for secret in secrets
                    ])

                    for secret, k8s_secret in zip(secrets, k8s_secrets):
                        secret_host_path = f'{job_root}/secrets/{k8s_secret.metadata.name}'

                        populate_secret_host_path(secret_host_path, k8s_secret.data)

                        mount_options.extend([
                            '-v', f'{secret_host_path}:{secret["mount_path"]}'
                        ])

                if j._mount_docker_socket:
                    mount_options.extend(['-v', '/var/run/docker.sock:/var/run/docker.sock'])

                main_cid, main_ok = await docker_run(
                    'docker', 'run', '-d',
                    *env_options, *mount_options, j._image, *j._command)
                print(f'{j._index}/main: {main_cid} {main_ok}')
            else:
                main_ok = False
                print(f'{j._index}/main: skipping: input failed')

            if j._output_files:
                if main_ok:
                    copy_script = 'set -ex\n'
                    for src, dest in j._input_files:
                        copy_script = copy_script + 'cp -a {src} {dest}\n'
                        output_cid, output_ok = await docker_run(
                            'docker', 'run', '-d', 'ubuntu:18.04', '-v', f'{root}/shared:/shared', '-v', f'{job_root}/io:/io', copy_script)
                        print(f'{j._index}/output: {output_cid} {output_ok}')
                else:
                    output_ok = False
                    print(f'{j._index}/output: skipping: main failed')
            else:
                output_ok = True

            j._succeeded = (input_ok and main_ok and output_ok)
            j._done = True


class Branch(Code):
    def __init__(self, owner, repo, branch, sha):
        self._owner = owner
        self._repo = repo
        self._branch = branch
        self._sha = sha

    def short_str(self):
        return f'br-{self._owner}-{self._repo}-{self._branch}'

    def repo_dir(self):
        return '.'

    def branch_url(self):
        return f'https://github.com/{self._owner}/{self._repo}'

    def config(self):
        return {
            'checkout_script': self.checkout_script(),
            'branch': self._branch,
            'repo': f'{self._owner}/{self._repo}',
            'repo_url': self.branch_url(),
            'sha': self._sha
        }

    def checkout_script(self):
        return f'''
{clone_or_fetch_script(self.branch_url())}

git checkout {shq(self._sha)}
'''

async def main():
    scope = 'deploy'
    code = Branch('cseed', 'hail', 'infra-1', 'dd8c84ee1601d9dd5643ec78fd9996cb51472e18')

    with open(f'build.yaml', 'r') as f:
        config = BuildConfiguration(code, f.read(), scope, requested_step_names=['deploy_batch'])

    token = generate_token()
    print(f'token {token}')
    batch = LocalBatchBuilder(
        attributes={
            'token': token
        }, callback=None)
    config.build(batch, code, scope)

    await batch.run(token)

asyncio.get_event_loop().run_until_complete(main())
