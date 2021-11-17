from typing import List, TYPE_CHECKING
import json
import logging
import asyncio
import aiohttp
import base64
import traceback

from hailtop.aiotools import BackgroundTaskManager
from hailtop.utils import time_msecs, Notice, retry_transient_errors
from hailtop import httpx
from gear import Database

from ..batch import batch_record_to_dict
from ..globals import complete_states, tasks, STATUS_FORMAT_VERSION
from ..batch_configuration import KUBERNETES_SERVER_URL
from ..batch_format_version import BatchFormatVersion
from ..spec_writer import SpecWriter
from ..file_store import FileStore
from ..instance_config import QuantifiedResource
from .instance import Instance

from .k8s_cache import K8sCache

if TYPE_CHECKING:
    from .instance_collection import InstanceCollectionManager  # pylint: disable=cyclic-import

log = logging.getLogger('job')


async def notify_batch_job_complete(db: Database, client_session: httpx.ClientSession, batch_id):
    record = await db.select_and_fetchone(
        '''
SELECT batches.*, SUM(`usage` * rate) AS cost
FROM batches
LEFT JOIN aggregated_batch_resources
  ON batches.id = aggregated_batch_resources.batch_id
LEFT JOIN resources
  ON aggregated_batch_resources.resource = resources.resource
WHERE id = %s AND NOT deleted AND callback IS NOT NULL AND
   batches.`state` = 'complete'
GROUP BY batches.id;
''',
        (batch_id,),
    )

    if not record:
        return
    callback = record['callback']

    log.info(f'making callback for batch {batch_id}: {callback}')

    async def request(session):
        await session.post(callback, json=batch_record_to_dict(record))
        log.info(f'callback for batch {batch_id} successful')

    try:
        if record['user'] == 'ci':
            # only jobs from CI may use batch's TLS identity
            await request(client_session)
        else:
            async with aiohttp.ClientSession(raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                await request(session)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception(f'callback for batch {batch_id} failed, will not retry.')


async def add_attempt_resources(db, batch_id, job_id, attempt_id, resources):
    if attempt_id:
        try:
            resource_args = [
                (batch_id, job_id, attempt_id, resource['name'], resource['quantity']) for resource in resources
            ]

            await db.execute_many(
                '''
INSERT INTO `attempt_resources` (batch_id, job_id, attempt_id, resource, quantity)
VALUES (%s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE quantity = quantity;
''',
                resource_args,
            )
        except Exception:
            log.exception(f'error while inserting resources for job {id}, attempt {attempt_id}')
            raise


async def mark_job_complete(
    app, batch_id, job_id, attempt_id, instance_name, new_state, status, start_time, end_time, reason, resources
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

    try:
        rv = await db.execute_and_fetchone(
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
        )
    except Exception:
        log.exception(f'error while marking job {id} complete on instance {instance_name}')
        raise

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

    await add_attempt_resources(db, batch_id, job_id, attempt_id, resources)

    if rv['rc'] != 0:
        log.info(f'mark_job_complete returned {rv} for job {id}')
        return

    old_state = rv['old_state']
    if old_state in complete_states:
        log.info(f'old_state {old_state} complete for job {id}, doing nothing')
        # already complete, do nothing
        return

    log.info(f'job {id} changed state: {rv["old_state"]} => {new_state}')

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
        )
    except Exception:
        log.info(f'error while marking job {id} started on {instance}')
        raise

    if rv['delta_cores_mcpu'] != 0 and instance.state == 'active':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])

    await add_attempt_resources(db, batch_id, job_id, attempt_id, resources)


async def mark_job_creating(app,
                            batch_id: int,
                            job_id: int,
                            attempt_id: str,
                            instance: Instance,
                            start_time: int,
                            resources: List[QuantifiedResource]):
    db: Database = app['db']

    id = (batch_id, job_id)

    log.info(f'mark job {id} creating')

    try:
        rv = await db.execute_and_fetchone(
            '''
CALL mark_job_creating(%s, %s, %s, %s, %s);
''',
            (batch_id, job_id, attempt_id, instance.name, start_time),
        )
    except Exception:
        log.info(f'error while marking job {id} creating on {instance}')
        raise

    log.info(rv)
    if rv['delta_cores_mcpu'] != 0 and instance.state == 'pending':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])

    await add_attempt_resources(db, batch_id, job_id, attempt_id, resources)


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

    db_spec = json.loads(record['spec'])

    if format_version.has_full_spec_in_gcs():
        job_spec = {
            'secrets': format_version.get_spec_secrets(db_spec),
            'service_account': format_version.get_spec_service_account(db_spec),
        }
    else:
        job_spec = db_spec

    job_spec['attempt_id'] = attempt_id

    userdata = json.loads(record['userdata'])

    secrets = job_spec.get('secrets', [])
    k8s_secrets = await asyncio.gather(
        *[
            k8s_cache.read_secret(secret['name'], secret['namespace'])
            for secret in secrets
        ]
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
        assert len(sa.secrets) == 1

        token_secret_name = sa.secrets[0].name

        secret = await k8s_cache.read_secret(token_secret_name, namespace)

        token = base64.b64decode(secret.data['token']).decode()
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
    token: {token}
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

    if format_version.has_full_spec_in_gcs():
        token, start_job_id = await SpecWriter.get_token_start_id(db, batch_id, job_id)
    else:
        token = None
        start_job_id = None

    return {
        'batch_id': batch_id,
        'job_id': job_id,
        'format_version': format_version.format_version,
        'token': token,
        'start_job_id': start_job_id,
        'user': record['user'],
        'gsa_key': gsa_key,
        'job_spec': job_spec,
    }


async def schedule_job(app, record, instance):
    assert instance.state == 'active'

    file_store: FileStore = app['file_store']
    db: Database = app['db']
    client_session: httpx.ClientSession = app['client_session']

    batch_id = record['batch_id']
    job_id = record['job_id']
    attempt_id = record['attempt_id']
    format_version = BatchFormatVersion(record['format_version'])

    id = (batch_id, job_id)

    try:
        try:
            body = await job_config(app, record, attempt_id)
        except Exception:
            log.exception('while making job config')
            status = {
                'version': STATUS_FORMAT_VERSION,
                'worker': None,
                'batch_id': batch_id,
                'job_id': job_id,
                'attempt_id': attempt_id,
                'user': record['user'],
                'state': 'error',
                'error': traceback.format_exc(),
                'container_statuses': {k: None for k in tasks},
            }

            if format_version.has_full_status_in_gcs():
                await file_store.write_status_file(batch_id, job_id, attempt_id, json.dumps(status))

            db_status = format_version.db_status(status)
            resources = []

            await mark_job_complete(
                app, batch_id, job_id, attempt_id, instance.name, 'Error', db_status, None, None, 'error', resources
            )
            raise

        log.info(f'schedule job {id} on {instance}: made job config')

        try:
            await client_session.post(
                f'http://{instance.ip_address}:5000/api/v1alpha/batches/jobs/create',
                json=body,
                timeout=aiohttp.ClientTimeout(total=2))
            await instance.mark_healthy()
        except aiohttp.ClientResponseError as e:
            await instance.mark_healthy()
            if e.status == 403:
                log.info(f'attempt already exists for job {id} on {instance}, aborting')
            if e.status == 503:
                log.info(f'job {id} cannot be scheduled because {instance} is shutting down, aborting')
            raise e
        except Exception:
            await instance.incr_failed_request_count()
            raise

        log.info(f'schedule job {id} on {instance}: called create job')

        rv = await db.execute_and_fetchone(
            '''
CALL schedule_job(%s, %s, %s, %s);
''',
            (batch_id, job_id, attempt_id, instance.name),
        )
    except Exception:
        log.exception(f'error while scheduling job {id} on {instance}')
        if instance.state == 'active':
            instance.adjust_free_cores_in_memory(record['cores_mcpu'])
        return

    if rv['delta_cores_mcpu'] != 0 and instance.state == 'active':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])

    log.info(f'schedule job {id} on {instance}: updated database')

    if rv['rc'] != 0:
        log.info(f'could not schedule job {id}, attempt {attempt_id} on {instance}, {rv}')
        return

    log.info(f'success scheduling job {id} on {instance}')
