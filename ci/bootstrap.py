from shlex import quote as shq

from ci.build import BuildConfiguration, Code
from ci.github import clone_or_fetch_script
from ci.utils import generate_token


class LocalJob:
    def __init__(self, image, command, **kwargs):
        self._image = image
        self._command = command
        self._kwargs = kwargs

        print(f'job: {image}, {command}, {kwargs}')

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
        job = LocalJob(image, command, **kwargs)
        self._jobs.append(job)
        return job

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
batch = LocalBatchBuilder(
    attributes={
        'token': token
    }, callback=None)
config.build(batch, code, scope)
