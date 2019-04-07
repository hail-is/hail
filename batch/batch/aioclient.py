import math
import time
import random
import yaml
import cerberus
import aiohttp

from . import schemas


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

    async def is_complete(self):
        if self._status:
            state = self._status['state']
            if state in ('Complete', 'Cancelled'):
                return True
        await self.status()
        state = self._status['state']
        return state in ('Complete', 'Cancelled')

    async def status(self):
        self._status = await self.client._get_job(self.id)
        return self._status

    async def wait(self):
        i = 0
        while True:
            if await self.is_complete():
                return self._status
            j = random.randrange(math.floor(1.1 ** i))
            time.sleep(0.100 * j)
            # max 4.45s
            if i < 64:
                i = i + 1

    async def cancel(self):
        await self.client._cancel_job(self.id)

    async def delete(self):
        await self.client._delete_job(self.id)

        # this object should not be referenced again
        del self.client
        del self.id
        del self.attributes
        del self.parent_ids
        del self.scratch_folder
        del self._status

    async def log(self):
        return await self.client._get_job_log(self.id)


class Batch:
    def __init__(self, client, id):
        self.client = client
        self.id = id

    async def create_job(self, image, command=None, args=None, env=None, ports=None,
                         resources=None, tolerations=None, volumes=None, security_context=None,
                         service_account_name=None, attributes=None, callback=None, parent_ids=None,
                         scratch_folder=None, input_files=None, output_files=None,
                         copy_service_account_name=None, always_run=False):
        if parent_ids is None:
            parent_ids = []
        return await self.client._create_job(
            image, command, args, env, ports, resources, tolerations, volumes, security_context,
            service_account_name, attributes, self.id, callback, parent_ids, scratch_folder,
            input_files, output_files, copy_service_account_name, always_run)

    async def close(self):
        await self.client._close_batch(self.id)

    async def status(self):
        return await self.client._get_batch(self.id)

    async def wait(self):
        i = 0
        while True:
            status = await self.status()
            if status['jobs']['Created'] == 0:
                return status
            j = random.randrange(math.floor(1.1 ** i))
            time.sleep(0.100 * j)
            # max 4.45s
            if i < 64:
                i = i + 1


class BatchClient:
    def __init__(self, url=None, timeout=60):
        if not url:
            url = 'http://batch.default'
        self.url = url
        self._session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=timeout))

    async def _create_job(self,  # pylint: disable=R0912
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
                          scratch_folder,
                          input_files,
                          output_files,
                          copy_service_account_name,
                          always_run):
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
            'spec': spec,
            'parent_ids': parent_ids,
            'always_run': always_run
        }
        if attributes:
            doc['attributes'] = attributes
        if batch_id:
            doc['batch_id'] = batch_id
        if callback:
            doc['callback'] = callback
        if scratch_folder:
            doc['scratch_folder'] = scratch_folder
        if input_files:
            doc['input_files'] = input_files
        if output_files:
            doc['output_files'] = output_files
        if copy_service_account_name:
            doc['copy_service_account_name'] = copy_service_account_name

        response = await self._session.post(self.url + '/jobs/create', json=doc)
        j = await response.json()

        return Job(self,
                   j['id'],
                   attributes=j.get('attributes'),
                   parent_ids=j.get('parent_ids', []))

    async def _get_job(self, id):
        response = await self._session.get(self.url + '/jobs/{}'.format(id))
        return await response.json()

    async def _get_job_log(self, id):
        response = await self._session.get(self.url + '/jobs/{}/log'.format(id))
        return await response.json()

    async def _delete_job(self, id):
        response = await self._session.delete(self.url + '/jobs/{}/delete'.format(id))
        return await response.json()

    async def _cancel_job(self, id):
        response = await self._session.post(self.url + '/jobs/{}/cancel'.format(id))
        return await response.json()

    async def _get_batch(self, batch_id):
        response = await self._session.get(self.url + '/batches/{}'.format(batch_id))
        return await response.json()

    async def _close_batch(self, batch_id):
        response = await self._session.post(self.url + '/batches/{}/close'.format(batch_id))
        return await response.json()

    async def _refresh_k8s_state(self):
        await self._session.post(self.url + '/refresh_k8s_state')

    async def list_jobs(self):
        response = await self._session.get(self.url + '/jobs')
        jobs = await response.json()
        return [Job(self,
                    j['id'],
                    attributes=j.get('attributes'),
                    parent_ids=j.get('parent_ids', []),
                    _status=j)
                for j in jobs]

    async def get_job(self, id):
        response = await self._session.get(self.url + '/jobs/{}'.format(id))
        j = await response.json()
        return Job(self,
                   j['id'],
                   attributes=j.get('attributes'),
                   parent_ids=j.get('parent_ids', []),
                   _status=j)

    async def create_job(self,
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
                         scratch_folder=None,
                         input_files=None,
                         output_files=None,
                         copy_service_account_name=None,
                         always_run=False):
        if parent_ids is None:
            parent_ids = []
        return await self._create_job(
            image, command, args, env, ports, resources, tolerations, volumes, security_context,
            service_account_name, attributes, None, callback, parent_ids, scratch_folder,
            input_files, output_files, copy_service_account_name, always_run)

    async def create_batch(self, attributes=None, callback=None, ttl=None):
        doc = {}
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if ttl:
            doc['ttl'] = ttl
        response = await self._session.post(self.url + '/batches/create', json=doc)
        batch = await response.json()
        return Batch(self, batch['id'])

    job_yaml_schema = {
        'spec': schemas.pod_spec,
        'type': {'type': 'string', 'allowed': ['execute']},
        'name': {'type': 'string'},
        'dependsOn': {'type': 'list', 'schema': {'type': 'string'}},
    }
    job_yaml_validator = cerberus.Validator(job_yaml_schema)

    async def create_batch_from_file(self, file):
        job_id_by_name = {}

        # pylint confused by f-strings
        def job_id_by_name_or_error(id, self_id):  # pylint: disable=unused-argument
            job = job_id_by_name.get(id)
            if job:
                return job
            raise ValueError(
                '"{self_id}" must appear in the file after its dependency "{id}"')

        batch = await self.create_batch()
        for doc in yaml.load(file):
            if not BatchClient.job_yaml_validator.validate(doc):
                raise BatchClient.job_yaml_validator.errors
            spec = doc['spec']
            type = doc['type']
            name = doc['name']
            dependsOn = doc.get('dependsOn', [])
            if type == 'execute':
                job = await batch.create_job(
                    parent_ids=[job_id_by_name_or_error(x, name) for x in dependsOn],
                    **spec)
                job_id_by_name[name] = job.id
        return batch

    async def close(self):
        await self._session.close()
        self._session = None
