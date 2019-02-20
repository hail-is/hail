import time
import random
import yaml

import cerberus

from . import api, schemas


class Job:
    def __init__(self, client, id, attributes=None, parent_ids=None, scratch_folder=None, _status=None):
        if parent_ids is None:
            parent_ids = []
        if attributes is None:
            attributes = {}

        self.client = client
        self.id = id
        self.attributes = attributes
        self.parent_ids = parent_ids
        self.scratch_folder = scratch_folder
        self._status = _status

    def is_complete(self):
        if self._status:
            state = self._status['state']
            if state in ('Complete', 'Cancelled'):
                return True
        return False

    def cached_status(self):
        assert self._status is not None
        return self._status

    def status(self):
        self._status = self.client._get_job(self.id)
        return self._status

    def wait(self):
        i = 0
        while True:
            self.status()  # update
            if self.is_complete():
                return self._status
            j = random.randrange(2 ** i)
            time.sleep(0.100 * j)
            # max 5.12s
            if i < 9:
                i = i + 1

    def cancel(self):
        self.client._cancel_job(self.id)

    def delete(self):
        self.client._delete_job(self.id)

        self.id = None
        self.attributes = None
        self._status = None

    def log(self):
        return self.client._get_job_log(self.id)


class Batch:
    def __init__(self, client, id):
        self.client = client
        self.id = id

    def create_job(self, image, command=None, args=None, env=None, ports=None,
                   resources=None, tolerations=None, volumes=None, security_context=None,
                   service_account_name=None, attributes=None, callback=None, parent_ids=None,
                   scratch_folder=None):
        if parent_ids is None:
            parent_ids = []
        return self.client._create_job(
            image, command, args, env, ports, resources, tolerations, volumes, security_context,
            service_account_name, attributes, self.id, callback, parent_ids, scratch_folder)

    def close(self):
        self.client._close_batch(self.id)

    def status(self):
        return self.client._get_batch(self.id)

    def wait(self):
        i = 0
        while True:
            status = self.status()
            if status['jobs']['Created'] == 0:
                return status
            j = random.randrange(2 ** i)
            time.sleep(0.100 * j)
            # max 5.12s
            if i < 9:
                i = i + 1


class BatchClient:
    def __init__(self, url=None, api=api.DEFAULT_API):
        if not url:
            url = 'http://batch.default'
        self.url = url
        self.api = api

    def _create_job(self,  # pylint: disable=R0912
                    image,
                    command,
                    args,
                    env,
                    ports,
                    resources,
                    tolerations,
                    volumes,
                    security_context,
                    service_account_name,
                    attributes,
                    batch_id,
                    callback,
                    parent_ids,
                    scratch_folder):
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
            'name': 'default'
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

        j = self.api.create_job(self.url, spec, attributes, batch_id, callback, parent_ids, scratch_folder)
        return Job(self, j['id'], j.get('attributes'), j.get('parent_ids', []))

    def _get_job(self, id):
        return self.api.get_job(self.url, id)

    def _get_job_log(self, id):
        return self.api.get_job_log(self.url, id)

    def _delete_job(self, id):
        self.api.delete_job(self.url, id)

    def _cancel_job(self, id):
        self.api.cancel_job(self.url, id)

    def _get_batch(self, batch_id):
        return self.api.get_batch(self.url, batch_id)

    def _close_batch(self, batch_id):
        return self.api.close_batch(self.url, batch_id)

    def _refresh_k8s_state(self):
        self.api.refresh_k8s_state(self.url)

    def list_jobs(self):
        jobs = self.api.list_jobs(self.url)
        return [Job(self, j['id'], j.get('attributes'), j.get('parent_ids', []), j) for j in jobs]

    def get_job(self, id):
        # make sure job exists
        j = self.api.get_job(self.url, id)
        return Job(self, j['id'], j.get('attributes'), j.get('parent_ids', []), j)

    def create_job(self,
                   image,
                   command=None,
                   args=None,
                   env=None,
                   ports=None,
                   resources=None,
                   tolerations=None,
                   volumes=None,
                   security_context=None,
                   service_account_name=None,
                   attributes=None,
                   callback=None,
                   parent_ids=None,
                   scratch_folder=None):
        if parent_ids is None:
            parent_ids = []
        return self._create_job(
            image, command, args, env, ports, resources, tolerations, volumes, security_context,
            service_account_name, attributes, None, callback, parent_ids, scratch_folder)

    def create_batch(self, attributes=None, callback=None, ttl=None):
        batch = self.api.create_batch(self.url, attributes, callback, ttl)
        return Batch(self, batch['id'])

    job_yaml_schema = {
        'spec': schemas.pod_spec,
        'type': {'type': 'string', 'allowed': ['execute']},
        'name': {'type': 'string'},
        'dependsOn': {'type': 'list', 'schema': {'type': 'string'}},
    }
    job_yaml_validator = cerberus.Validator(job_yaml_schema)

    def create_batch_from_file(self, file):
        job_id_by_name = {}

        def job_id_by_name_or_error(id, self_id):
            job = job_id_by_name.get(id)
            if job:
                return job
            raise ValueError(
                '"{self_id}" must appear in the file after its dependency "{id}"')

        batch = self.create_batch()
        for doc in yaml.load(file):
            if not BatchClient.job_yaml_validator.validate(doc):
                raise BatchClient.job_yaml_validator.errors
            spec = doc['spec']
            type = doc['type']
            name = doc['name']
            dependsOn = doc.get('dependsOn', [])
            if type == 'execute':
                job = batch.create_job(
                    parent_ids=[job_id_by_name_or_error(x, name) for x in dependsOn],
                    **spec)
                job_id_by_name[name] = job.id
        return batch
