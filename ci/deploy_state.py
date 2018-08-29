class Deploying(object):
    def __init__(self, job):
        self.job = job

    def __str__(self):
        return f'deploying {self.job.id}'

    def to_json(self):
        return {
            'type': 'Deploying',
            'job_id': self.job.id
        }

    def __eq__(self, other):
        return (
            isinstance(other, Deploying) and
            self.job.id == other.job.id
        )

    def __ne__(self, other):
        return not self == other


class DeployJobFailure(object):
    def __init__(self, exit_code):
        self.exit_code = exit_code

    def __str__(self):
        return f'deploy failed {self.exit_code}'

    def to_json(self):
        return {
            'type': 'DeployJobFailure',
            'exit_code': self.exit_code
        }

    def __eq__(self, other):
        return (
          isinstance(other, DeployJobFailure) and
          self.exit_code == other.exit_code
        )

    def __ne__(self, other):
        return not self == other


class DeployOtherFailure(object):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f'deploy failed {self.message}'

    def to_json(self):
        return {
            'type': 'DeployOtherFailure'
        }

    def __eq__(self, other):
        return (
          isinstance(other, DeployOtherFailure) and
          self.exit_code == other.exit_code
        )

    def __ne__(self, other):
        return not self == other
