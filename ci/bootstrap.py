import os
from shlex import quote as shq

from ci.build import BuildConfiguration, Code
from ci.github import clone_or_fetch_script
from ci.utils import generate_token


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

    def run(self):
        os.makedirs(f'_/shared')
        
        for j in self._jobs:
            if j._parents:
                for p in j._parents:
                    assert p._done

            os.makedirs(f'_/{j._index}/secrets')

            # localize secrets
            # copy inputs
            # copy outputs

            mount_options = [
                '-v', '_/shared:/shared'
            ]
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

scope = 'deploy'
code = Branch('cseed', 'hail', 'infra-1', '04cbbf10928aa88ee8be30b65c80388801cdcd32')

with open(f'build.yaml', 'r') as f:
    config = BuildConfiguration(code, f.read(), scope)

token = generate_token()
print(f'token {token}')
batch = LocalBatchBuilder(
    attributes={
        'token': token
    }, callback=None)
config.build(batch, code, scope)

batch.run()
