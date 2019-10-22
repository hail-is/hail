import json
import logging
import asyncio
import aiohttp

from .globals import states, complete_states, valid_state_transitions, tasks
from .log_store import LogStore

log = logging.getLogger('batch')


class JobStateWriteFailure(Exception):
    pass


class Job:
    async def _create_pod(self):
        assert self.userdata is not None
        assert self._state in states
        assert self._state == 'Running'

        # FIXME handle exceptions?
        await self.app['driver'].create_pod(
            name=self._pod_name,
            batch_id=self.batch_id,
            job_spec=self._spec,
            userdata=self.userdata,
            output_directory=self.directory)

    async def _delete_pod(self):
        await self.app['driver'].delete_pod(name=self._pod_name)

    async def _read_logs(self):
        if self._state in ('Pending', 'Cancelled'):
            return None

        if self._state == 'Running':
            return await self.app['driver'].read_pod_logs(self._pod_name)

        async def _read_log_from_gcs(task_name):
            pod_log = await self.app['log_store'].read_gs_file(LogStore.container_log_path(self.directory, task_name))
            return task_name, pod_log

        assert self._state in ('Error', 'Failed', 'Success')
        future_logs = asyncio.gather(*[_read_log_from_gcs(task) for task in tasks])
        return {k: v for k, v in await future_logs}

    async def _read_pod_status(self):
        if self._state in ('Pending', 'Cancelled'):
            return None

        if self._state == 'Running':
            return await self.app['driver'].read_pod_status(self._pod_name)

        return await self.app['log_store'].read_gs_file(LogStore.pod_status_path(self.directory))

    async def _delete_gs_files(self):
        await self.app['log_store'].delete_gs_files(self.directory)

    @staticmethod
    def from_record(app, record):
        if not record:
            return None

        userdata = json.loads(record['userdata'])
        spec = json.loads(record['spec'])
        status = json.loads(record['status']) if record['status'] else None

        return Job(app, batch_id=record['batch_id'], job_id=record['job_id'],
                   userdata=userdata, user=record['user'],
                   status=status, state=record['state'],
                   cancelled=record['cancelled'], directory=record['directory'],
                   spec=spec)

    @staticmethod
    async def from_pod(app, pod_status):
        batch_id = pod_status['batch_id']
        job_id = pod_status['job_id']
        user = pod_status['user']
        return await Job.from_db(app, batch_id, job_id, user)

    @staticmethod
    async def from_db(app, batch_id, job_id, user):
        jobs = await Job.from_db_multiple(app, batch_id, job_id, user)
        if len(jobs) == 1:
            return jobs[0]
        return None

    @staticmethod
    async def from_db_multiple(app, batch_id, job_ids, user):
        records = await app['db'].jobs.get_undeleted_records(batch_id, job_ids, user)
        jobs = [Job.from_record(app, record) for record in records]
        return jobs

    def __init__(self, app, batch_id, job_id, userdata, user,
                 status, state, cancelled, directory,
                 spec):
        self.app = app
        self.batch_id = batch_id
        self.job_id = job_id
        self.id = (batch_id, job_id)

        self.userdata = userdata
        self.user = user
        self.status = status
        self.directory = directory

        name = f'batch-{batch_id}-job-{job_id}'
        self._pod_name = name
        self._state = state
        self._cancelled = cancelled
        self._spec = spec

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

    async def refresh_parents_and_maybe_create(self):
        for record in await self.app['db'].jobs.get_parents(*self.id):
            parent_job = Job.from_record(self.app, record)
            assert parent_job.batch_id == self.batch_id
            await self.parent_new_state(parent_job._state, *parent_job.id)

    async def set_state(self, new_state):
        assert new_state in valid_state_transitions[self._state], f'{self._state} -> {new_state}'
        if self._state != new_state:
            n_updated = await self.app['db'].jobs.update_record(*self.id, compare_items={'state': self._state}, state=new_state)
            if n_updated == 0:
                log.warning(f'changing the state from {self._state} -> {new_state} '
                            f'for job {self.id} failed due to the expected state not in db')
                raise JobStateWriteFailure()

            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))
            self._state = new_state
            await self.notify_children(new_state)

    async def notify_children(self, new_state):
        if new_state not in complete_states:
            return

        children = [Job.from_record(self.app, record) for record in await self.app['db'].jobs.get_children(*self.id)]
        for child in children:
            await child.parent_new_state(new_state, *self.id)

    async def parent_new_state(self, new_state, parent_batch_id, parent_job_id):
        del parent_job_id
        assert parent_batch_id == self.batch_id
        if new_state in complete_states:
            await self.create_if_ready()

    async def create_if_ready(self):
        incomplete_parent_ids = await self.app['db'].jobs.get_incomplete_parents(*self.id)
        if self._state == 'Pending' and not incomplete_parent_ids:
            await self.set_state('Running')
            parents = [Job.from_record(self.app, record) for record in await self.app['db'].jobs.get_parents(*self.id)]
            if (self.always_run or
                    (not self._cancelled and all(p.is_successful() for p in parents))):
                log.info(f'all parents complete for {self.id},'
                         f' creating pod')
                await self._create_pod()
            else:
                log.info(f'parents deleted, cancelled, or failed: cancelling {self.id}')
                await self.set_state('Cancelled')

    async def cancel(self):
        self._cancelled = True

        if not self.always_run and self._state == 'Running':
            await self.set_state('Cancelled')  # must call before deleting resources to prevent race conditions
            await self._delete_pod()

    def is_complete(self):
        return self._state in complete_states

    def is_successful(self):
        return self._state == 'Success'

    async def mark_unscheduled(self):
        updated_job = await Job.from_db(self.app, *self.id, self.user)
        if updated_job.is_complete():
            log.info(f'job is already completed in db, not rescheduling pod')
            return

        await self._delete_pod()
        if self._state == 'Running' and (not self._cancelled or self.always_run):
            await self._create_pod()

    async def mark_complete(self, status):
        pod_state = status['state']
        if pod_state == 'succeeded':
            new_state = 'Success'
        elif pod_state == 'error':
            new_state = 'Error'
        else:
            assert pod_state == 'failed', pod_state
            new_state = 'Failed'

        n_updated = await self.app['db'].jobs.update_record(*self.id,
                                                            compare_items={'state': self._state},
                                                            status=json.dumps(status),
                                                            state=new_state)
        if n_updated == 0:
            log.info(f'could not update job {self.id} due to db not matching expected state')
            raise JobStateWriteFailure()

        self.status = status

        if self._state != new_state:
            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))

        self._state = new_state

        await self._delete_pod()
        await self.notify_children(new_state)

        log.info('job {} complete with state {}, status {}'.format(self.id, self._state, status))

        if self.batch_id:
            batch = await Batch.from_db(self.app, self.batch_id, self.user)
            if batch is not None:
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
                # self.status['container_statuses'][k]['container_status']['exit_code']
                k: getopt(getopt(self.status['container_statuses'][k], 'container_status'), 'exit_code') for
                k in tasks
            }
            result['duration'] = {k: getopt(self.status['container_statuses']['k']['timing'], 'runtime') for k in tasks}
            result['message'] = {
                # the execution of the pod container might have
                # failed, or the docker container might have completed
                # (wait returned) but had status error
                # (self.status['container_statuses'][k]['error'] or
                #  self.status['container_statuses'][k]['container_status']['error'])
                k: (getopt(self.status['container_statuses'][k], 'error') or
                    getopt(getopt(self.status['container_statuses'][k], 'container_status'), 'error'))
                for k in tasks
            }
        if self.attributes:
            result['attributes'] = self.attributes
        return result


class Batch:
    @staticmethod
    def from_record(app, record, deleted=False):
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

            return Batch(app,
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
    async def from_db(app, ids, user):
        batches = await Batch.from_db_multiple(app, ids, user)
        if len(batches) == 1:
            return batches[0]
        return None

    @staticmethod
    async def from_db_multiple(app, ids, user):
        records = await app['db'].batch.get_undeleted_records(ids, user)
        batches = [Batch.from_record(app, record) for record in records]
        return batches

    @staticmethod
    async def create_batch(app, attributes, callback, userdata, n_jobs):
        user = userdata['username']

        id = await app['db'].batch.new_record(
            attributes=json.dumps(attributes),
            callback=callback,
            userdata=json.dumps(userdata),
            user=user,
            deleted=False,
            cancelled=False,
            closed=False,
            n_jobs=n_jobs)

        batch = Batch(app, id=id, attributes=attributes, callback=callback,
                      userdata=userdata, user=user, state='running',
                      complete=False, deleted=False, cancelled=False,
                      closed=False)

        if attributes is not None:
            items = [{'batch_id': id, 'key': k, 'value': v} for k, v in attributes.items()]
            success = await app['db'].batch_attributes.new_records(items)
            if not success:
                await batch.delete()
                return

        return batch

    def __init__(self, app, id, attributes, callback, userdata, user,
                 state, complete, deleted, cancelled, closed):
        self.app = app
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
        jobs = await self.app['db'].jobs.get_records_by_batch(self.id, limit, offset)
        return [Job.from_record(self.app, record) for record in jobs]

    # called by driver
    async def _cancel_jobs(self):
        await asyncio.gather(*[j.cancel() for j in await self.get_jobs()])

    # called by front end
    async def cancel(self):
        await self.app['db'].batch.update_record(self.id, cancelled=True, closed=True)
        self.cancelled = True
        self.closed = True
        log.info(f'batch {self.id} cancelled')

    # called by driver
    async def _close_jobs(self):
        await asyncio.gather(*[j._create_pod() for j in await self.get_jobs()
                               if j._state == 'Running'])

    # called by front end
    async def close(self):
        await self.app['db'].batch.update_record(self.id, closed=True)
        self.closed = True
        log.info(f'batch {self.id} closed')

    # called by driver
    # FIXME make called by front end
    async def mark_deleted(self):
        await self.cancel()
        await self.app['db'].batch.update_record(self.id,
                                                 deleted=True)
        self.deleted = True
        self.closed = True
        log.info(f'batch {self.id} marked for deletion')

    async def delete(self):
        # Job deleted from database when batch is deleted with delete cascade
        await asyncio.gather(*[j._delete_gs_files() for j in await self.get_jobs()])
        await self.app['db'].batch.delete_record(self.id)
        log.info(f'batch {self.id} deleted')

    async def mark_job_complete(self):
        if self.complete and self.callback:
            log.info(f'making callback for batch {self.id}: {self.callback}')
            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                    await session.post(self.callback, json=await self.to_dict(include_jobs=False))
                    log.info(f'callback for batch {self.id} successful')
            except Exception:  # pylint: disable=broad-except
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
