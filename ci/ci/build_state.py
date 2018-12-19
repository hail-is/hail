import json

import re
from batch.client import Job

from .batch_helper import try_to_cancel_job
from .ci_logging import log
from .environment import batch_client, CONTEXT


def build_state_from_gh_json(d):
    assert isinstance(d, list), d
    assert all([isinstance(x, dict) for x in d]), d
    my_statuses = [status for status in d if status['context'] == CONTEXT]
    if len(my_statuses) != 0:
        latest_status = my_statuses[0]
        state = latest_status['state']
        assert state in [
            'pending', 'failure', 'success'
        ], state  # 'error' is allowed by github but not used by me
        description = latest_status['description']
        try:
            matches = re.findall(r'({.*})$', description)
            assert len(matches) == 1, f'{d} {matches}'
            doc = json.loads(matches[0])
        except Exception as e:
            log.exception(
                'could not parse build state from description {latest_status}')
            return Unknown()

        return build_state_from_json(doc)
    else:
        return Unknown()


def build_state_from_json(d):
    t = d['type']
    if t == 'Merged':
        return Merged(d['target_sha'])
    elif t == 'Mergeable':
        return Mergeable(d['target_sha'])
    elif t == 'Failure':
        return Failure(d['exit_code'], d['image'], d['target_sha'])
    elif t == 'NoMergeSHA':
        return NoMergeSHA(d['exit_code'], d['target_sha'])
    elif t == 'Building':
        return Building(
            batch_client.get_job(d['job_id']),
            d['image'],
            d['target_sha'])
    elif t == 'Buildable':
        return Buildable(d['image'], d['target_sha'])
    else:
        assert t == 'Unknown'
        return Unknown()


class Merged(object):
    def __init__(self, target_sha):
        self.target_sha = target_sha

    def transition(self, other):
        raise ValueError(f'bad transition {self} to {other}')

    def __str__(self):
        return f'merged'

    def to_json(self):
        return {
            'type': 'Merged',
            'target_sha': self.target_sha
        }

    def gh_state(self):
        return 'success'

    def __eq__(self, other):
        return (isinstance(other, Merged) and
                self.target_sha == other.target_sha)

    def __ne__(self, other):
        return not self == other


class Mergeable(object):
    def __init__(self, target_sha):
        self.target_sha = target_sha

    def transition(self, other):
        if not isinstance(other, Merged):
            log.warning(
                f'usually Mergeable should go to Merged, but going to {other}'
            )
        return other

    def __str__(self):
        return f'successful build'

    def to_json(self):
        return {
            'type': 'Mergeable',
            'target_sha': self.target_sha
        }

    def gh_state(self):
        return 'success'

    def __eq__(self, other):
        return (isinstance(other, Mergeable) and
                self.target_sha == other.target_sha)

    def __ne__(self, other):
        return not self == other


class Failure(object):
    def __init__(self, exit_code, image, target_sha):
        self.exit_code = exit_code
        self.image = image
        self.target_sha = target_sha

    def retry(self, job):
        return Building(job, self.image, self.target_sha)

    def transition(self, other):
        return other

    def __str__(self):
        return f'failing build {self.exit_code}'

    def to_json(self):
        return {
            'type': 'Failure',
            'exit_code': self.exit_code,
            'image': self.image,
            'target_sha': self.target_sha
        }

    def gh_state(self):
        return 'failure'

    def __eq__(self, other):
        return (isinstance(other, Failure) and
                self.exit_code == other.exit_code and
                self.image == other.image and
                self.target_sha == other.target_sha)

    def __ne__(self, other):
        return not self == other


class NoMergeSHA(object):
    def __init__(self, exit_code, target_sha):
        self.exit_code = exit_code
        self.target_sha = target_sha

    def retry(self, job, image):
        return Building(job, image, self.target_sha)

    def transition(self, other):
        return other

    def __str__(self):
        return f'could not find merge sha in last build {self.exit_code}'

    def to_json(self):
        return {
            'type': 'NoMergeSHA',
            'exit_code': self.exit_code,
            'target_sha': self.target_sha
        }

    def gh_state(self):
        return 'failure'

    def __eq__(self, other):
        return (isinstance(other,
                           NoMergeSHA) and self.exit_code == other.exit_code
                and self.target_sha == other.target_sha)

    def __ne__(self, other):
        return not self == other


class Building(object):
    def __init__(self, job, image, target_sha):
        assert isinstance(job, Job)
        self.job = job
        self.image = image
        self.target_sha = target_sha

    def success(self, merged_sha):
        return Mergeable(merged_sha, self.target_sha)

    def failure(self, exit_code):
        return Failure(exit_code, self.image, self.target_sha)

    def no_merge_sha(self, exit_code):
        return NoMergeSHA(exit_code, self.target_sha)

    def transition(self, other):
        if isinstance(other, Merged):
            raise ValueError(f'bad transition {self} to {other}')

        if (not isinstance(other, Failure) and
            not isinstance(other, Mergeable) and
            not isinstance(other, NoMergeSHA)):
            log.info(f'cancelling unneeded job {self.job.id} {self} {other}')
            try_to_cancel_job(self.job)
        return other

    def __str__(self):
        return f'build {self.job.id} pending. target: {self.target_sha[0:12]}'

    def to_json(self):
        return {
            'type': 'Building',
            'job_id': self.job.id,
            'image': self.image,
            'target_sha': self.target_sha
        }

    def gh_state(self):
        return 'pending'

    def __eq__(self, other):
        return (isinstance(other,
                           Building) and self.job.id == other.job.id
                and self.image == other.image
                and self.target_sha == other.target_sha)

    def __ne__(self, other):
        return not self == other


class Buildable(object):
    def __init__(self, image, target_sha):
        self.image = image
        self.target_sha = target_sha

    def building(self, job_id):
        return Building(job_id, self.image, self.target_sha)

    def transition(self, other):
        if (not isinstance(other, Building) and
            not isinstance(other, Buildable)):
            log.warning(f'unusual transition {self} to {other}')
        return other

    def __str__(self):
        return f'build not yet started'

    def to_json(self):
        return {
            'type': 'Buildable',
            'image': self.image,
            'target_sha': self.target_sha
        }

    def gh_state(self):
        return 'pending'

    def __eq__(self, other):
        return (isinstance(other, Buildable) and self.image == other.image
                and self.target_sha == other.target_sha)

    def __ne__(self, other):
        return not self == other


class NoImage(object):
    def __init__(self, target_sha):
        self.target_sha = target_sha

    def transition(self, other):
        if (not isinstance(other, Buildable) and
            not isinstance(other, Building) and
            not (isinstance(other, NoImage) and self != other)):
            raise ValueError(f'bad transition {self} to {other}')
        return other

    def __str__(self):
        return f'no hail-ci-build-image found {self.target_sha[:12]}'

    def to_json(self):
        return {'type': 'NoImage', 'target_sha': self.target_sha}

    def gh_state(self):
        return 'failure'

    def __eq__(self, other):
        return (
            isinstance(other, NoImage) and
            self.target_sha == other.target_sha
        )

    def __ne__(self, other):
        return not self == other


class Unknown(object):
    def __init__(self):
        pass

    def buildable(self, image):
        return Buildable(image)

    def transition(self, other):
        return other

    def __str__(self):
        return 'unknown build state'

    def to_json(self):
        return {'type': 'Unknown'}

    def gh_state(self):
        raise ValueError('do not use Unknown to update github')

    def __eq__(self, other):
        return isinstance(other, Unknown)

    def __ne__(self, other):
        return not self == other
