import os
from shlex import quote as shq
import asyncio

import kubernetes_asyncio as kube

from ci.build import BuildConfiguration, Code
from ci.github import clone_or_fetch_script
from ci.utils import generate_token

from batch.driver.k8s_cache import K8sCache


class LocalJob:
    def __init__(self, index, image, command, *,
                 env=None, mount_docker_socket=False, secrets=None, parents=None,
                 input_files=None, output_files=None,
                 **kwargs):
        self._index = index
        self._image = image
        self._command = command

        self._env = env
        self._mount_docker_socket = mount_docker_socket
        self._parents = parents
        self._secrets = secrets
        self._input_files = input_files
        self._output_files = output_files
        self._kwargs = kwargs

        self._done = False

        print(f'job: {image}, {command}, {env}, {mount_docker_socket}, {secrets}, {parents}, {input_files}, {output_files}, {kwargs}')


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

    async def run(self):
        await kube.config.load_kube_config()
        k8s_client = kube.client.CoreV1Api()
        k8s_cache = K8sCache(k8s_client, refresh_time=5)
        
        os.makedirs(f'_/shared')
        
        for j in self._jobs:
            if j._parents:
                for p in j._parents:
                    assert p._done

            secrets_host_path = f'_/{j._index}/secrets'
            os.makedirs(secrets_host_path)

            # localize secrets
            # copy inputs
            # copy outputs

            mount_options = [
                '-v', '_/shared:/shared'
            ]

            secrets = j._secrets
            if secrets:
                print(secrets)
                k8s_secrets = await asyncio.gather(*[
                    k8s_cache.read_secret(
                        secret['name'], secret['namespace'],
                        5)
                    for secret in secrets
                ])
                
                for k8s_secret in k8s_secrets:
                    secret_host_path = f'{secrets_host_path}/{secret["name"]}'

                    populate_secret_host_path(secret_host_path, k8s_secret['data'])

                    mount_options.extend([
                        '-v', f'{secrets_host_path}:{secret["mount_path"]}'
                    ])

            if j._mount_docker_socket:
                mount_options.extend(['-v', '/var/run/docker.sock:/var/run/docker.sock'])

            docker_cmd = [
                'docker',
                'run',
                *mount_options,
                j._image,
                *[shq(c) for c in j._command]
            ]
            
            print(docker_cmd)
            
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
    code = Branch('cseed', 'hail', 'infra-1', '04cbbf10928aa88ee8be30b65c80388801cdcd32')

    with open(f'build.yaml', 'r') as f:
        config = BuildConfiguration(code, f.read(), scope, requested_step_names=['deploy_batch'])

    token = generate_token()
    print(f'token {token}')
    batch = LocalBatchBuilder(
        attributes={
            'token': token
        }, callback=None)
    config.build(batch, code, scope)

    await batch.run()

asyncio.get_event_loop().run_until_complete(main())
