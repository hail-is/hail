import asyncio
import base64
import collections
import json
import logging
import traceback
from typing import TYPE_CHECKING, Dict, List

import aiohttp

from gear import Database, K8sCache, transaction
from hailtop import httpx
from hailtop.aiotools import BackgroundTaskManager
from hailtop.utils import Notice, retry_transient_errors, time_msecs

from ..batch import job_group_record_to_dict
from ..batch_configuration import KUBERNETES_SERVER_URL
from ..batch_format_version import BatchFormatVersion
from ..file_store import FileStore
from ..globals import STATUS_FORMAT_VERSION, complete_states, tasks
from ..instance_config import QuantifiedResource
from ..spec_writer import SpecWriter
from .instance import Instance

if TYPE_CHECKING:
    from .instance_collection import InstanceCollectionManager  # pylint: disable=cyclic-import

log = logging.getLogger('job')


async def notify_batch_job_complete(db: Database, client_session: httpx.ClientSession, batch_id):
    records = db.select_and_fetchall(
        '''
SELECT job_groups.*, batches.`user`, batches.billing_project, batches.msec_mcpu, COALESCE(SUM(`usage` * rate), 0) AS cost,
  CAST(COALESCE(states.n_completed, 0) AS SIGNED) AS n_completed, CAST(COALESCE(states.n_succeeded, 0) AS SIGNED) AS n_succeeded,
  CAST(COALESCE(states.n_failed, 0) AS SIGNED) AS n_failed, CAST(COALESCE(states.n_cancelled, 0) AS SIGNED) AS n_cancelled,
  batches_cancelled.id IS NOT NULL AS cancelled
FROM job_groups
LEFT JOIN batches ON job_groups.batch_id = batches.id
LEFT JOIN batches_cancelled
    ON job_groups.batch_id = batches_cancelled.id AND job_groups.job_group_id = batches_cancelled.job_group_id
LEFT JOIN (
  SELECT id, job_group_id,
    CAST(COALESCE(SUM(batches_n_jobs_in_complete_states.n_completed), 0) AS SIGNED) AS n_completed,
    CAST(COALESCE(SUM(batches_n_jobs_in_complete_states.n_succeeded), 0) AS SIGNED) AS n_succeeded,
    CAST(COALESCE(SUM(batches_n_jobs_in_complete_states.n_failed), 0) AS SIGNED) AS n_failed,
    CAST(COALESCE(SUM(batches_n_jobs_in_complete_states.n_cancelled), 0) AS SIGNED) AS n_cancelled
  FROM batches_n_jobs_in_complete_states
  WHERE id = %s
  GROUP BY id, job_group_id
) AS states ON job_groups.batch_id = states.id AND job_groups.job_group_id = states.job_group_id
LEFT JOIN (
  SELECT batch_id, job_group_id, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_batch_resources_v2
  WHERE batch_id = %s
  GROUP BY batch_id, job_group_id, resource_id
) AS ajgr
  ON job_groups.batch_id = ajgr.batch_id AND job_groups.job_group_id = ajgr.job_group_id
LEFT JOIN resources
  ON ajgr.resource_id = resources.resource_id
WHERE batches.id = %s AND NOT deleted AND job_groups.batch_id = %s AND job_groups.callback IS NOT NULL AND job_groups.`state` = 'complete'
GROUP BY job_groups.batch_id, job_groups.job_group_id;
''',
        (batch_id, batch_id, batch_id, batch_id),
        'notify_batch_job_group_complete',
    )

    async def request(session, callback, data, job_group_path):
        await session.post(callback, json=data)
        log.info(f'callback for batch {batch_id} job group {job_group_path} successful')

    async for record in records:
        callback = record['callback']
        data = job_group_record_to_dict(record)
        job_group_path = record['path']

        try:
            if record['user'] == 'ci':
                # only jobs from CI may use batch's TLS identity
                await request(client_session, callback, data, job_group_path)
            else:
                async with httpx.client_session() as session:
                    await request(session, callback, data, job_group_path)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.info(f'callback for batch {batch_id} job group {job_group_path} failed, will not retry.')


async def add_attempt_resources(app, db, batch_id, job_id, attempt_id, resources: List[QuantifiedResource]):
    resource_name_to_id = app['resource_name_to_id']
    if attempt_id and len(resources) > 0:
        try:
            _resources: Dict[str, int] = collections.defaultdict(lambda: 0)
            for resource in resources:
                _resources[resource['name']] += resource['quantity']

            # This must be sorted in order to match the order of values in the actual SQL table!
            _resources = dict(sorted(_resources.items()))

            resource_args = [
                (
                    batch_id,
                    job_id,
                    attempt_id,
                    resource_name_to_id[name].resource_id,
                    resource_name_to_id[name].deduped_resource_id,
                    quantity,
                )
                for name, quantity in _resources.items()
            ]

            await db.execute_many(
                '''
INSERT INTO `attempt_resources` (batch_id, job_id, attempt_id, resource_id, deduped_resource_id, quantity)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE quantity = quantity;
''',
                resource_args,
                'add_attempt_resources',
            )
        except Exception:
            log.exception(f'error while inserting resources for job {job_id}, attempt {attempt_id}')
            raise


async def mark_job_complete(
    app,
    batch_id,
    job_id,
    job_group_id,
    attempt_id,
    instance_name,
    new_state,
    status,
    start_time,
    end_time,
    reason,
    resources: List[QuantifiedResource],
    *,
    marked_job_started=False,
):
    scheduler_state_changed: Notice = app['scheduler_state_changed']
    cancel_ready_state_changed: asyncio.Event = app['cancel_ready_state_changed']
    db: Database = app['db']
    client_session: httpx.ClientSession = app['client_session']

    inst_coll_manager: 'InstanceCollectionManager' = app['driver'].inst_coll_manager
    task_manager: BackgroundTaskManager = app['task_manager']

    id = (batch_id, job_id)

    log.info(f'marking job {id} complete new_state {new_state}')

    now = time_msecs()

    @transaction(db)
    async def _mark_complete(tx):
        try:
            rv = await tx.execute_and_fetchone(
                'CALL mark_job_complete(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);',
                (
                    batch_id,
                    job_id,
                    attempt_id,
                    instance_name,
                    new_state,
                    json.dumps(status) if status is not None else None,
                    start_time,
                    end_time,
                    reason,
                    now,
                ),
                'mark_job_complete',
            )
        except Exception:
            log.exception(f'error while marking job {id} complete on instance {instance_name}')
            raise

        try:
            complete_job_groups = tx.execute_and_fetchall(
                '''
SELECT job_group_parents.batch_id, job_group_parents.parent_id,
  job_groups.n_jobs, CAST(COALESCE(SUM(n_completed), 0) AS SIGNED) AS n_completed
FROM batches_n_jobs_in_complete_states
INNER JOIN job_group_parents ON batches_n_jobs_in_complete_states.id = job_group_parents.batch_id AND
  batches_n_jobs_in_complete_states.job_group_id = job_group_parents.parent_id
LEFT JOIN job_groups ON job_group_parents.batch_id = job_groups.batch_id AND job_group_parents.parent_id = job_groups.job_group_id
WHERE job_group_parents.batch_id = %s AND job_group_parents.job_group_id = %s
GROUP BY id, parent_id, n_jobs
HAVING n_jobs = n_completed;
''',
                (batch_id, job_group_id),
                'get_complete_job_groups',
            )
            job_groups = [(now, record['batch_id'], record['parent_id']) async for record in complete_job_groups]

            if job_groups:
                await tx.execute_many(
                    '''
UPDATE job_groups
SET state = 'complete', time_completed = %s
WHERE batch_id = %s AND job_group_id = %s;
''',
                    job_groups,
                    'update_job_group_state',
                )
        except Exception:
            log.exception(f'error while getting and marking completed job groups for job {id}')
            raise

        return rv

    rv = await _mark_complete()  # pylint: disable=no-value-for-parameter

    scheduler_state_changed.notify()
    cancel_ready_state_changed.set()

    instance = None

    if instance_name:
        instance = inst_coll_manager.get_instance(instance_name)
        if instance:
            if rv['delta_cores_mcpu'] != 0 and instance.state == 'active':
                # may also create scheduling opportunities, set above
                instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])
        else:
            log.warning(f'mark_complete for job {id} from unknown {instance}')

    if not marked_job_started:
        await add_attempt_resources(app, db, batch_id, job_id, attempt_id, resources)

    if rv['rc'] != 0:
        log.info(f'mark_job_complete returned {rv} for job {id}')
        return

    old_state = rv['old_state']
    if old_state in complete_states:
        log.info(f'old_state {old_state} complete for job {id}, doing nothing')
        # already complete, do nothing
        return

    await notify_batch_job_complete(db, client_session, batch_id)

    if instance and not instance.inst_coll.is_pool and instance.state == 'active':
        task_manager.ensure_future(instance.kill())


async def mark_job_started(app, batch_id, job_id, attempt_id, instance, start_time, resources):
    db: Database = app['db']

    id = (batch_id, job_id)

    log.info(f'mark job {id} started')

    try:
        rv = await db.execute_and_fetchone(
            '''
CALL mark_job_started(%s, %s, %s, %s, %s);
''',
            (batch_id, job_id, attempt_id, instance.name, start_time),
            'mark_job_started',
        )
    except Exception:
        log.info(f'error while marking job {id} started on {instance}')
        raise

    if rv['delta_cores_mcpu'] != 0 and instance.state == 'active':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])

    await add_attempt_resources(app, db, batch_id, job_id, attempt_id, resources)


async def mark_job_creating(
    app,
    batch_id: int,
    job_id: int,
    attempt_id: str,
    instance: Instance,
    start_time: int,
    resources: List[QuantifiedResource],
):
    db: Database = app['db']

    id = (batch_id, job_id)

    log.info(f'mark job {id} creating')

    try:
        rv = await db.execute_and_fetchone(
            '''
CALL mark_job_creating(%s, %s, %s, %s, %s);
''',
            (batch_id, job_id, attempt_id, instance.name, start_time),
            'mark_job_creating',
        )
    except Exception:
        log.info(f'error while marking job {id} creating on {instance}')
        raise

    log.info(rv)
    if rv['delta_cores_mcpu'] != 0 and instance.state == 'pending':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])

    await add_attempt_resources(app, db, batch_id, job_id, attempt_id, resources)


async def unschedule_job(app, record):
    cancel_ready_state_changed: asyncio.Event = app['cancel_ready_state_changed']
    scheduler_state_changed: Notice = app['scheduler_state_changed']
    db: Database = app['db']
    client_session: httpx.ClientSession = app['client_session']
    inst_coll_manager = app['driver'].inst_coll_manager

    batch_id = record['batch_id']
    job_id = record['job_id']
    attempt_id = record['attempt_id']
    id = (batch_id, job_id)

    instance_name = record['instance_name']
    assert instance_name is not None

    log.info(f'unscheduling job {id}, attempt {attempt_id} from instance {instance_name}')

    end_time = time_msecs()

    try:
        rv = await db.execute_and_fetchone(
            'CALL unschedule_job(%s, %s, %s, %s, %s, %s);',
            (batch_id, job_id, attempt_id, instance_name, end_time, 'cancelled'),
        )
    except Exception:
        log.exception(f'error while unscheduling job {id} on instance {instance_name}')
        raise

    log.info(f'unschedule job {id}: updated database {rv}')

    # job that was running is now ready to be cancelled
    cancel_ready_state_changed.set()

    instance = inst_coll_manager.get_instance(instance_name)
    if not instance:
        log.warning(f'unschedule job {id}, attempt {attempt_id}: unknown instance {instance_name}')
        return

    if rv['delta_cores_mcpu'] and instance.state == 'active':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])
        scheduler_state_changed.notify()
        log.info(f'unschedule job {id}, attempt {attempt_id}: updated {instance} free cores')

    url = f'http://{instance.ip_address}:5000/api/v1alpha/batches/{batch_id}/jobs/{job_id}/delete'

    async def make_request():
        if instance.state in ('inactive', 'deleted'):
            return
        try:
            await client_session.delete(url)
            await instance.mark_healthy()
        except asyncio.TimeoutError:
            await instance.incr_failed_request_count()
            return
        except aiohttp.ClientResponseError as err:
            if err.status == 404:
                await instance.mark_healthy()
                return
            await instance.incr_failed_request_count()
            raise

    await retry_transient_errors(make_request)

    if not instance.inst_coll.is_pool:
        await instance.kill()

    log.info(f'unschedule job {id}, attempt {attempt_id}: called delete job')


async def job_config(app, record, attempt_id):
    k8s_cache: K8sCache = app['k8s_cache']
    db: Database = app['db']

    format_version = BatchFormatVersion(record['format_version'])
    batch_id = record['batch_id']
    job_id = record['job_id']
    job_group_id = record['job_group_id']

    db_spec = json.loads(record['spec'])

    if format_version.has_full_spec_in_cloud():
        job_spec = {
            'secrets': format_version.get_spec_secrets(db_spec),
            'service_account': format_version.get_spec_service_account(db_spec),
        }
    else:
        job_spec = db_spec

    job_spec['job_group_id'] = job_group_id
    job_spec['attempt_id'] = attempt_id

    userdata = json.loads(record['userdata'])

    secrets = job_spec.get('secrets', [])
    k8s_secrets = await asyncio.gather(
        *[k8s_cache.read_secret(secret['name'], secret['namespace']) for secret in secrets]
    )

    gsa_key = None

    # backwards compatibility
    gsa_key_secret_name = userdata.get('hail_credentials_secret_name') or userdata['gsa_key_secret_name']

    for secret, k8s_secret in zip(secrets, k8s_secrets):
        if secret['name'] == gsa_key_secret_name:
            gsa_key = k8s_secret.data
        secret['data'] = k8s_secret.data

    assert gsa_key

    service_account = job_spec.get('service_account')
    if service_account:
        namespace = service_account['namespace']
        name = service_account['name']

        sa = await k8s_cache.read_service_account(name, namespace)
        if sa.secrets is not None:
            # ServiceAccounts created prior to Kubernetes 1.24 have autogenerated secrets
            assert len(sa.secrets) == 1
            token_secret_name = sa.secrets[0].name
        else:
            # ServiceAccounts post v1.24 don't have autogenerated secrets and we make those ourselves
            token_secret_name = f'{name}-token'

        secret = await k8s_cache.read_secret(token_secret_name, namespace)

        user_token = base64.b64decode(secret.data['token']).decode()
        cert = secret.data['ca.crt']

        kube_config = f'''
apiVersion: v1
clusters:
- cluster:
    certificate-authority: /.kube/ca.crt
    server: {KUBERNETES_SERVER_URL}
  name: default-cluster
contexts:
- context:
    cluster: default-cluster
    user: {namespace}-{name}
    namespace: {namespace}
  name: default-context
current-context: default-context
kind: Config
preferences: {{}}
users:
- name: {namespace}-{name}
  user:
    token: {user_token}
'''

        job_spec['secrets'].append(
            {
                'name': 'kube-config',
                'mount_path': '/.kube',
                'data': {'config': base64.b64encode(kube_config.encode()).decode(), 'ca.crt': cert},
            }
        )

        env = job_spec.get('env')
        if not env:
            env = []
            job_spec['env'] = env
        env.append({'name': 'KUBECONFIG', 'value': '/.kube/config'})

    if format_version.has_full_spec_in_cloud():
        spec_token, start_job_id = await SpecWriter.get_token_start_id(db, batch_id, job_id)
    else:
        spec_token = None
        start_job_id = None

    return {
        'batch_id': batch_id,
        'job_id': job_id,
        'format_version': format_version.format_version,
        'token': spec_token,
        'start_job_id': start_job_id,
        'user': record['user'],
        'gsa_key': gsa_key,
        'job_spec': job_spec,
        'queue_time': time_msecs() - record['time_ready'] if record['time_ready'] else None,
    }


async def mark_job_errored(app, batch_id, job_id, job_group_id, attempt_id, user, format_version, error_msg):
    file_store: FileStore = app['file_store']

    status = {
        'version': STATUS_FORMAT_VERSION,
        'worker': None,
        'batch_id': batch_id,
        'job_id': job_id,
        'attempt_id': attempt_id,
        'user': user,
        'state': 'error',
        'error': error_msg,
        'container_statuses': {k: None for k in tasks},
    }

    if format_version.has_full_status_in_gcs():
        await file_store.write_status_file(batch_id, job_id, attempt_id, json.dumps(status))

    db_status = format_version.db_status(status)

    await mark_job_complete(
        app, batch_id, job_id, job_group_id, attempt_id, None, 'Error', db_status, None, None, 'error', []
    )


async def schedule_job(app, record, instance):
    assert instance.state == 'active'

    db: Database = app['db']
    client_session: httpx.ClientSession = app['client_session']

    batch_id = record['batch_id']
    job_id = record['job_id']
    job_group_id = record['job_group_id']
    attempt_id = record['attempt_id']
    format_version = BatchFormatVersion(record['format_version'])

    id = (batch_id, job_id)

    try:
        body = await job_config(app, record, attempt_id)
    except Exception:
        log.exception(f'while making job config for job {id} with attempt id {attempt_id}')

        await mark_job_errored(
            app, batch_id, job_id, job_group_id, attempt_id, record['user'], format_version, traceback.format_exc()
        )
        raise

    try:
        await client_session.post(
            f'http://{instance.ip_address}:5000/api/v1alpha/batches/jobs/create',
            json=body,
            timeout=aiohttp.ClientTimeout(total=2),
        )
        await instance.mark_healthy()
    except aiohttp.ClientResponseError as e:
        await instance.mark_healthy()
        if e.status == 403:
            log.info(f'attempt {attempt_id} already exists for job {id} on {instance}, aborting')
        if e.status == 503:
            log.info(f'job {id} attempt {attempt_id} cannot be scheduled because {instance} is shutting down, aborting')
        raise e
    except Exception:
        await instance.incr_failed_request_count()
        raise

    try:
        rv = await db.execute_and_fetchone(
            '''
CALL schedule_job(%s, %s, %s, %s);
''',
            (batch_id, job_id, attempt_id, instance.name),
            'schedule_job',
        )
    except Exception:
        log.exception(f'Error while running schedule_job procedure for job {id} attempt {attempt_id}')
        raise

    if rv['delta_cores_mcpu'] != 0 and instance.state == 'active':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])

    if rv['rc'] != 0:
        log.info(f'could not schedule job {id}, attempt {attempt_id} on {instance} in the db, {rv}')
        return

    log.info(f'success scheduling job {id} on {instance}')
