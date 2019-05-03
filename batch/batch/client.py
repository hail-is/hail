import os

import hailjwt as hj

from . import api
from .poll_until import poll_until


class Job:
    def __init__(self, client, batch, id, attributes=None, parent_ids=None, _status=None):
        if parent_ids is None:
            parent_ids = []
        if attributes is None:
            attributes = {}

        self.client = client
        self.batch = batch
        self.id = id
        self.attributes = attributes
        self.parent_ids = parent_ids
        self._status = _status

    def is_complete(self):
        if self.batch._is_running:
            raise ValueError("Job is not running yet")

        if self._status:
            state = self._status['state']
            if state in ('Complete', 'Cancelled'):
                return True
        return False

    def cached_status(self):
        if self.batch._is_running:
            raise ValueError("Job is not running yet")

        assert self._status is not None
        return self._status

    def status(self):
        if self.batch._is_running:
            raise ValueError("Job is not running yet")

        self._status = self.client._get_job(self.batch.id, self.id)
        return self._status

    def wait(self):
        if self.batch._is_running:
            raise ValueError("Job is not running yet")

        def update_and_is_complete():
            self.status()
            return self.is_complete()
        poll_until(update_and_is_complete)
        return self._status

    def log(self):
        if self.batch._is_running:
            raise ValueError("Job is not running yet")
        return self.client._get_job_log(self.batch.id, self.id)


class Batch:
    def __init__(self, client, id, attributes, callback, _is_running=False):
        self.client = client
        self.id = id
        self.attributes = attributes
        self.callback = callback
        self._is_running = _is_running
        self._job_idx = 0
        self._job_docs = []

    def create_job(self, image, command=None, args=None, env=None, ports=None,
                   resources=None, tolerations=None, volumes=None, security_context=None,
                   service_account_name=None, attributes=None, callback=None, parents=None,
                   input_files=None, output_files=None, always_run=False):
        self._job_idx += 1

        if not self._is_running:
            raise ValueError("Cannot add a job to a batch that has already been run")

        if parents is None:
            parents = []

        parent_ids = [p.id for p in parents]

        def _is_valid_parent(p):
            return p.batch == self and 0 < p.id < self._job_idx

        if not all([_is_valid_parent(p) for p in parents]):
            raise ValueError("Found the following invalid parent ids:\n{}".format(
                "\n  ".join([str((p.batch.id, p.id)) for p in parents if not _is_valid_parent(p)])))

        if env:
            env = [{'name': k, 'value': v} for (k, v) in env.items()]
        else:
            env = []
        env.extend([{
            'name': 'POD_IP',
            'valueFrom': {
                'fieldRef': {'fieldPath': 'status.podIP'}
            }
        }, {
            'name': 'POD_NAME',
            'valueFrom': {
                'fieldRef': {'fieldPath': 'metadata.name'}
            }
        }])

        container = {
            'image': image,
            'name': 'main'
        }
        if command:
            container['command'] = command
        if args:
            container['args'] = args
        if env:
            container['env'] = env
        if ports:
            container['ports'] = [{
                'containerPort': p,
                'protocol': 'TCP'
            } for p in ports]
        if resources:
            container['resources'] = resources
        if volumes:
            container['volumeMounts'] = [v['volume_mount'] for v in volumes]
        spec = {
            'containers': [container],
            'restartPolicy': 'Never'
        }
        if volumes:
            spec['volumes'] = [v['volume'] for v in volumes]
        if tolerations:
            spec['tolerations'] = tolerations
        if security_context:
            spec['securityContext'] = security_context
        if service_account_name:
            spec['serviceAccountName'] = service_account_name

        doc = {
            'job_id': self._job_idx,
            'spec': spec,
            'parent_ids': parent_ids,
            'always_run': always_run
        }
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if input_files:
            doc['input_files'] = input_files
        if output_files:
            doc['output_files'] = output_files

        self._job_docs.append(doc)

        j = Job(self.client,
                None,
                self._job_idx,
                attributes=attributes,
                parent_ids=parent_ids)
        return j

    def status(self):
        if self._is_running:
            raise ValueError("Batch is not running yet")
        return self.client._get_batch(self.id)

    def wait(self):
        if self._is_running:
            raise ValueError("Batch is not running yet")

        def update_and_is_complete():
            status = self.status()
            if status['complete']:
                return status
            return False
        return poll_until(update_and_is_complete)

    def cancel(self):
        if self._is_running:
            raise ValueError("Batch is not running yet")
        self.client._cancel_batch(self.id)

    def delete(self):
        if self._is_running:
            raise ValueError("Batch is not running yet")
        self.client._delete_batch(self.id)

    def get_job(self, job_id):
        if self._is_running:
            raise ValueError("Batch is not running yet")

        if not 0 < job_id <= self._job_idx:
            raise ValueError(f"Invalid job id {job_id} for batch {self.id} with {self._job_idx} jobs")

        j = self.client._get_job(self.id, job_id)
        return Job(self.client,
                   self,
                   j['job_id'],
                   attributes=j.get('attributes'),
                   parent_ids=j.get('parent_ids', []),
                   _status=j)

    def run(self):
        if not self._is_running:
            raise ValueError("Batch is already running")

        doc = {}
        if self.attributes:
            doc['attributes'] = self.attributes
        if self.callback:
            doc['callback'] = self.callback
        doc['jobs'] = self._job_docs
        b = self.client._create_batch(doc)

        self.id = b.id
        self.attributes = b.attributes
        self.callback = b.callback
        self._is_running = False


class BatchClient:
    def __init__(self, url=None, timeout=None, token_file=None, token=None, headers=None):
        if token_file is not None and token is not None:
            raise ValueError('set only one of token_file and token')
        if not url:
            url = 'http://batch.default'
        self.url = url
        if token is None:
            token_file = (token_file or
                          os.environ.get('HAIL_TOKEN_FILE') or
                          os.path.expanduser('~/.hail/token'))
            if not os.path.exists(token_file):
                raise ValueError(
                    f'cannot create a client without a token. no file was '
                    f'found at {token_file}')
            with open(token_file) as f:
                token = f.read()
        userdata = hj.JWTClient.unsafe_decode(token)
        assert "bucket_name" in userdata
        self.bucket = userdata["bucket_name"]
        self.api = api.API(timeout=timeout,
                           cookies={'user': token},
                           headers=headers)

    def _get_job(self, batch_id, job_id):
        return self.api.get_job(self.url, batch_id, job_id)

    def _get_job_log(self, batch_id, job_id):
        return self.api.get_job_log(self.url, batch_id, job_id)

    def _get_batch(self, batch_id):
        return self.api.get_batch(self.url, batch_id)

    def _cancel_batch(self, batch_id):
        self.api.cancel_batch(self.url, batch_id)

    def _delete_batch(self, batch_id):
        self.api.delete_batch(self.url, batch_id)

    def _create_batch(self, doc):
        b = self.api.create_batch(self.url, doc)
        return Batch(self, b['id'], b.get('attributes'), b.get('callback'))

    def list_batches(self, complete=None, success=None, attributes=None):
        batches = self.api.list_batches(self.url, complete=complete, success=success, attributes=attributes)
        return [Batch(self,
                      b['id'],
                      attributes=b.get('attributes'),
                      callback=b.get('callback'))
                for b in batches]

    def get_batch(self, id):
        b = self._get_batch(id)
        return Batch(self,
                     b['id'],
                     b.get('attributes'),
                     b.get('callback'))

    def create_batch(self, attributes=None, callback=None):
        return Batch(self,
                     id=None,
                     attributes=attributes,
                     callback=callback,
                     _is_running=True)
