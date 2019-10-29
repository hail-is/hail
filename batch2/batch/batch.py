import json
import logging
import asyncio
import aiohttp
from hailtop.utils import request_retry_transient_errors

from .globals import tasks
from .database import check_call_procedure
from .log_store import LogStore

log = logging.getLogger('batch')


class Job:
    async def _read_logs(self, log_store, inst_pool):
        if self._state in ('Pending', 'Cancelled'):
            return None

        if self._state == 'Running':
            instance = None
            if self.instance_id is not None:
                instance = inst_pool.instances.get(self.instance_id)

            if instance is None:
                return None
            assert instance.ip_address

            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                url = f'http://{instance.ip_address}:5000/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/logs'
                resp = await request_retry_transient_errors(session, 'GET', url)
                return await resp.json()

        async def _read_log_from_gcs(task):
            log = await log_store.read_gs_file(LogStore.container_log_path(self.directory, task))
            return task, log

        assert self._state in ('Error', 'Failed', 'Success')
        future_logs = asyncio.gather(*[_read_log_from_gcs(task) for task in tasks])
        return {k: v for k, v in await future_logs}

    async def _read_status(self, log_store, inst_pool):
        if self._state in ('Pending', 'Cancelled'):
            return None

        if self._state == 'Running':
            instance = None
            if self.instance_id is not None:
                instance = inst_pool.instances.get(self.instance_id)

            if instance is None:
                return None
            assert instance.ip_address

            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                url = f'http://{instance.ip_address}:5000/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/status'
                resp = request_retry_transient_errors(session, 'GET', url)
                return await resp.json()

        return await log_store.read_gs_file(LogStore.job_status_path(self.directory))

    @staticmethod
    def from_record(db, record):
        if not record:
            return None

        userdata = json.loads(record['userdata'])
        spec = json.loads(record['spec'])
        status = json.loads(record['status']) if record['status'] else None

        return Job(db, batch_id=record['batch_id'], job_id=record['job_id'],
                   userdata=userdata, user=record['user'],
                   instance_id=record['instance_id'], status=status, state=record['state'],
                   cancelled=record['cancelled'], directory=record['directory'],
                   spec=spec, cores_mcpu=record['cores_mcpu'])

    @staticmethod
    async def from_db(db, batch_id, job_id, user):
        jobs = await Job.from_db_multiple(db, batch_id, job_id, user)
        if len(jobs) == 1:
            return jobs[0]
        return None

    @staticmethod
    async def from_db_multiple(db, batch_id, job_ids, user):
        records = await db.jobs.get_undeleted_records(batch_id, job_ids, user)
        jobs = [Job.from_record(db, record) for record in records]
        return jobs

    def __init__(self, db, batch_id, job_id, userdata, user,
                 instance_id, status, state, cancelled, directory, spec, cores_mcpu):
        self.db = db
        self.batch_id = batch_id
        self.job_id = job_id
        self.id = (batch_id, job_id)

        self.userdata = userdata
        self.user = user
        self.instance_id = instance_id
        self.status = status
        self.directory = directory

        self._state = state
        self._cancelled = cancelled
        self._spec = spec
        self.cores_mcpu = cores_mcpu

    @property
    def attributes(self):
        return self._spec.get('attributes')

    @property
    def input_files(self):
        return self._spec.get('input_files')

    @property
    def output_files(self):
        return self._spec.get('output_files')

    @property
    def always_run(self):
        return self._spec.get('always_run', False)

    @property
    def pvc_size(self):
        return self._spec.get('pvc_size')

    # FIXME move
    async def mark_complete(self, scheduler_state_changed, inst_pool, status):
        status_state = status['state']
        if status_state == 'succeeded':
            new_state = 'Success'
        elif status_state == 'error':
            new_state = 'Error'
        else:
            assert status_state == 'failed', status_state
            new_state = 'Failed'

        rv = await check_call_procedure(
            self.db.pool,
            'CALL mark_job_complete(%s, %s, %s, %s, %s);',
            (self.batch_id, self.job_id, new_state, json.dumps(status)))

        # update instance
        instance_id = rv['instance_id']
        instance = inst_pool.id_intance.get(instance_id)
        if instance:
            inst_pool.adjust_for_remove_instance(instance)
            instance.free_cores_mcpu -= self.cores_mcpu
            inst_pool.adjust_for_add_instance(instance)

        scheduler_state_changed.set()

        if self._state != new_state:
            log.info(f'{self} changed state: {self._state} -> {new_state}')
        self._state = new_state
        self.status = status
        self.instance_id = None

        log.info(f'{self} complete with state {self._state}, status {status}')

        batch = await Batch.from_db(self.db, self.batch_id, self.user)
        if batch:
            await batch.mark_job_complete()

    def to_dict(self):
        def getopt(obj, attr):
            if obj is None:
                return None
            return obj.get(attr)

        result = {
            'batch_id': self.batch_id,
            'job_id': self.job_id,
            'state': self._state
        }
        # FIXME can't change this yet, batch and batch2 share client
        if self.status:
            if 'error' in self.status:
                result['error'] = self.status['error']
            result['exit_code'] = {
                k: getopt(getopt(self.status['container_statuses'][k], 'container_status'), 'exit_code') for
                k in tasks
            }
            result['duration'] = {k: getopt(self.status['container_statuses'][k]['timing'], 'runtime') for k in tasks}
            result['message'] = {
                # the execution of the job container might have
                # failed, or the docker container might have completed
                # (wait returned) but had status error
                k: (getopt(self.status['container_statuses'][k], 'error') or
                    getopt(getopt(self.status['container_statuses'][k], 'container_status'), 'error'))
                for k in tasks
            }
        if self.attributes:
            result['attributes'] = self.attributes
        return result

    def __str__(self):
        return f'job ({self.batch_id}, {self.job_id})'


class Batch:
    @staticmethod
    def from_record(db, record, deleted=False):
        if record is not None:
            if not deleted:
                assert not record['deleted']
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])

            if record['n_failed'] > 0:
                state = 'failure'
            elif record['n_cancelled'] > 0:
                state = 'cancelled'
            elif record['closed'] and record['n_succeeded'] == record['n_jobs']:
                state = 'success'
            else:
                state = 'running'

            complete = record['closed'] and record['n_completed'] == record['n_jobs']

            return Batch(db,
                         id=record['id'],
                         attributes=attributes,
                         callback=record['callback'],
                         userdata=userdata,
                         user=record['user'],
                         state=state,
                         complete=complete,
                         deleted=record['deleted'],
                         cancelled=record['cancelled'],
                         closed=record['closed'])
        return None

    @staticmethod
    async def from_db(db, ids, user):
        batches = await Batch.from_db_multiple(db, ids, user)
        if len(batches) == 1:
            return batches[0]
        return None

    @staticmethod
    async def from_db_multiple(db, ids, user):
        records = await db.batch.get_undeleted_records(ids, user)
        batches = [Batch.from_record(db, record) for record in records]
        return batches

    @staticmethod
    async def create_batch(db, attributes, callback, userdata, n_jobs):
        user = userdata['username']

        id = await db.batch.new_record(
            attributes=json.dumps(attributes),
            callback=callback,
            userdata=json.dumps(userdata),
            user=user,
            deleted=False,
            cancelled=False,
            closed=False,
            n_jobs=n_jobs)

        batch = Batch(db, id=id, attributes=attributes, callback=callback,
                      userdata=userdata, user=user, state='running',
                      complete=False, deleted=False, cancelled=False,
                      closed=False)

        if attributes is not None:
            items = [{'batch_id': id, 'key': k, 'value': v} for k, v in attributes.items()]
            await db.batch_attributes.new_records(items)

        return batch

    def __init__(self, db, id, attributes, callback, userdata, user,
                 state, complete, deleted, cancelled, closed):
        self.db = db
        self.id = id
        self.attributes = attributes
        self.callback = callback
        self.userdata = userdata
        self.user = user
        self.state = state
        self.complete = complete
        self.deleted = deleted
        self.cancelled = cancelled
        self.closed = closed

    async def get_jobs(self, limit=None, offset=None):
        jobs = await self.db.jobs.get_records_by_batch(self.id, limit, offset)
        return [Job.from_record(self.db, record) for record in jobs]

    # FIXME move these to front-end, directly use database
    # called by front end
    async def cancel(self):
        await self.db.batch.update_record(self.id, cancelled=True)
        self.cancelled = True
        log.info(f'{self} cancelled')

    # called by front end
    async def close(self):
        await check_call_procedure(
            self.db.pool, 'CALL close_batch(%s);', (self.id,))
        self.closed = True
        log.info(f'{self} closed')

    # called by front end
    async def mark_deleted(self):
        await self.cancel()
        await self.db.batch.update_record(self.id, cancelled=True, deleted=True)
        self.cancelled = True
        self.deleted = True
        log.info(f'{self} marked for deletion')

    async def mark_job_complete(self):
        if self.complete and self.callback:
            log.info(f'making callback for batch {self.id}: {self.callback}')
            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                    await session.post(self.callback, json=await self.to_dict(include_jobs=False))
                    log.info(f'callback for batch {self.id} successful')
            except Exception:
                log.exception(f'callback for batch {self.id} failed, will not retry.')

    def is_complete(self):
        return self.complete

    def is_successful(self):
        return self.state == 'success'

    async def to_dict(self, include_jobs=False, limit=None, offset=None):
        result = {
            'id': self.id,
            'state': self.state,
            'complete': self.complete,
            'closed': self.closed
        }
        if self.attributes:
            result['attributes'] = self.attributes
        if include_jobs:
            jobs = await self.get_jobs(limit, offset)
            result['jobs'] = sorted([j.to_dict() for j in jobs], key=lambda j: j['job_id'])
        return result

    def __str__(self):
        return f'batch {self.id}'
