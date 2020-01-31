import json
import logging
import asyncio
import aiohttp
import base64
import traceback
from hailtop.utils import time_msecs, sleep_and_backoff, is_transient_error

from .globals import complete_states, tasks
from .batch_configuration import KUBERNETES_TIMEOUT_IN_SECONDS, \
    KUBERNETES_SERVER_URL
from .utils import cost_from_msec_mcpu
from .batch_format_version import BatchFormatVersion
from .spec_writer import SpecWriter

log = logging.getLogger('batch')


def batch_record_to_dict(app, record):
    if record['state'] == 'open':
        state = 'open'
    elif record['n_failed'] > 0:
        state = 'failure'
    elif record['cancelled'] or record['n_cancelled'] > 0:
        state = 'cancelled'
    elif record['state'] == 'complete':
        assert record['n_succeeded'] == record['n_jobs']
        state = 'success'
    else:
        state = 'running'

    d = {
        'id': record['id'],
        'billing_project': record['billing_project'],
        'state': state,
        'complete': record['state'] == 'complete',
        'closed': record['state'] != 'open',
        'n_jobs': record['n_jobs'],
        'n_completed': record['n_completed'],
        'n_succeeded': record['n_succeeded'],
        'n_failed': record['n_failed'],
        'n_cancelled': record['n_cancelled']
    }

    attributes = json.loads(record['attributes'])
    if attributes:
        d['attributes'] = attributes

    msec_mcpu = record['msec_mcpu']
    d['msec_mcpu'] = msec_mcpu

    cost = cost_from_msec_mcpu(app, msec_mcpu)
    d['cost'] = f'${cost:.4f}'

    return d


async def notify_batch_job_complete(app, db, batch_id):
    record = await db.select_and_fetchone(
        '''
SELECT *
FROM batches
WHERE id = %s AND NOT deleted AND callback IS NOT NULL AND
   batches.`state` = 'complete'
''',
        (batch_id,))

    if not record:
        return
    callback = record['callback']

    log.info(f'making callback for batch {batch_id}: {callback}')

    try:
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
            await session.post(callback, json=batch_record_to_dict(app, record))
            log.info(f'callback for batch {batch_id} successful')
    except Exception:
        log.exception(f'callback for batch {batch_id} failed, will not retry.')


async def mark_job_complete(app, batch_id, job_id, attempt_id, instance_name, new_state,
                            status, start_time, end_time, reason):
    scheduler_state_changed = app['scheduler_state_changed']
    cancel_ready_state_changed = app['cancel_ready_state_changed']
    db = app['db']
    inst_pool = app['inst_pool']

    id = (batch_id, job_id)

    log.info(f'marking job {id} complete new_state {new_state}')

    now = time_msecs()

    try:
        rv = await db.execute_and_fetchone(
            'CALL mark_job_complete(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);',
            (batch_id, job_id, attempt_id, instance_name, new_state,
             json.dumps(status) if status is not None else None,
             start_time, end_time, reason, now))
    except Exception:
        log.exception(f'error while marking job {id} complete on instance {instance_name}')
        raise

    scheduler_state_changed.set()
    cancel_ready_state_changed.set()

    if instance_name:
        instance = inst_pool.name_instance.get(instance_name)
        if instance:
            if rv['delta_cores_mcpu'] != 0 and instance.state == 'active':
                # may also create scheduling opportunities, set above
                instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])
        else:
            log.warning(f'mark_complete for job {id} from unknown {instance}')

    if rv['rc'] != 0:
        log.info(f'mark_job_complete returned {rv} for job {id}')
        return

    old_state = rv['old_state']
    if old_state in complete_states:
        log.info(f'old_state {old_state} complete for job {id}, doing nothing')
        # already complete, do nothing
        return

    log.info(f'job {id} changed state: {rv["old_state"]} => {new_state}')

    await notify_batch_job_complete(app, db, batch_id)


async def mark_job_started(app, batch_id, job_id, attempt_id, instance, start_time):
    db = app['db']

    id = (batch_id, job_id)

    log.info(f'mark job {id} started')

    try:
        rv = await db.execute_and_fetchone(
            '''
    CALL mark_job_started(%s, %s, %s, %s, %s);
    ''',
            (batch_id, job_id, attempt_id, instance.name, start_time))
    except Exception:
        log.exception(f'error while marking job {id} started on {instance}')
        raise

    if rv['delta_cores_mcpu'] != 0 and instance.state == 'active':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])


def job_record_to_dict(app, record, name):
    format_version = BatchFormatVersion(record['format_version'])

    db_status = record['status']
    if db_status:
        db_status = json.loads(db_status)
        exit_code, duration = format_version.get_status_exit_code_duration(db_status)
    else:
        exit_code = None
        duration = None

    result = {
        'batch_id': record['batch_id'],
        'job_id': record['job_id'],
        'name': name,
        'state': record['state'],
        'exit_code': exit_code,
        'duration': duration
    }

    msec_mcpu = record['msec_mcpu']
    result['msec_mcpu'] = msec_mcpu

    cost = cost_from_msec_mcpu(app, msec_mcpu)
    result['cost'] = f'${cost:.4f}'

    return result


async def unschedule_job(app, record):
    cancel_ready_state_changed = app['cancel_ready_state_changed']
    scheduler_state_changed = app['scheduler_state_changed']
    db = app['db']
    inst_pool = app['inst_pool']

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
            (batch_id, job_id, attempt_id, instance_name, end_time, 'cancelled'))
    except Exception:
        log.exception(f'error while unscheduling job {id} on instance {instance_name}')
        raise

    log.info(f'unschedule job {id}: updated database {rv}')

    # job that was running is now ready to be cancelled
    cancel_ready_state_changed.set()

    instance = inst_pool.name_instance.get(instance_name)
    if not instance:
        log.warning(f'unschedule job {id}, attempt {attempt_id}: unknown instance {instance_name}')
        return

    if rv['delta_cores_mcpu'] and instance.state == 'active':
        instance.adjust_free_cores_in_memory(rv['delta_cores_mcpu'])
        scheduler_state_changed.set()
        log.info(f'unschedule job {id}, attempt {attempt_id}: updated {instance} free cores')

    url = (f'http://{instance.ip_address}:5000'
           f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/delete')
    delay = 0.1
    while True:
        if instance.state in ('inactive', 'deleted'):
            break
        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                await session.delete(url)
                await instance.mark_healthy()
                break
        except Exception as e:
            if (isinstance(e, aiohttp.ClientResponseError) and
                    e.status == 404):  # pylint: disable=no-member
                await instance.mark_healthy()
                break
            else:
                await instance.incr_failed_request_count()
                if is_transient_error(e):
                    pass
                else:
                    raise
        delay = await sleep_and_backoff(delay)

    log.info(f'unschedule job {id}, attempt {attempt_id}: called delete job')


async def job_config(app, record, attempt_id):
    k8s_cache = app['k8s_cache']
    db = app['db']

    format_version = BatchFormatVersion(record['format_version'])
    batch_id = record['batch_id']
    job_id = record['job_id']

    db_spec = json.loads(record['spec'])

    if format_version.has_full_spec_in_gcs():
        job_spec = {
            'secrets': format_version.get_spec_secrets(db_spec),
            'service_account': format_version.get_spec_service_account(db_spec)
        }
    else:
        job_spec = db_spec

    job_spec['attempt_id'] = attempt_id

    userdata = json.loads(record['userdata'])

    secrets = job_spec.get('secrets', [])
    k8s_secrets = await asyncio.gather(*[
        k8s_cache.read_secret(
            secret['name'], secret['namespace'],
            KUBERNETES_TIMEOUT_IN_SECONDS)
        for secret in secrets
    ])

    gsa_key = None
    for secret, k8s_secret in zip(secrets, k8s_secrets):
        if secret['name'] == userdata['gsa_key_secret_name']:
            gsa_key = k8s_secret.data
        secret['data'] = k8s_secret.data

    assert gsa_key

    service_account = job_spec.get('service_account')
    if service_account:
        namespace = service_account['namespace']
        name = service_account['name']

        sa = await k8s_cache.read_service_account(
            name, namespace, KUBERNETES_TIMEOUT_IN_SECONDS)
        assert len(sa.secrets) == 1

        token_secret_name = sa.secrets[0].name

        secret = await k8s_cache.read_secret(
            token_secret_name, namespace, KUBERNETES_TIMEOUT_IN_SECONDS)

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

        job_spec['secrets'].append({
            'name': 'kube-config',
            'mount_path': '/.kube',
            'data': {'config': base64.b64encode(kube_config.encode()).decode(),
                     'ca.crt': cert}
        })

        env = job_spec.get('env')
        if not env:
            env = []
            job_spec['env'] = env
        env.append({'name': 'KUBECONFIG',
                    'value': '/.kube/config'})

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
        'job_spec': job_spec
    }


async def schedule_job(app, record, instance):
    assert instance.state == 'active'

    log_store = app['log_store']
    db = app['db']

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
                'worker': None,
                'batch_id': batch_id,
                'job_id': job_id,
                'attempt_id': attempt_id,
                'user': record['user'],
                'state': 'error',
                'error': traceback.format_exc(),
                'container_statuses': {k: {} for k in tasks}
            }

            if format_version.has_full_status_in_gcs():
                await log_store.write_status_file(batch_id, job_id, attempt_id, json.dumps(status))

            db_status = format_version.db_status(status)

            await mark_job_complete(app, batch_id, job_id, attempt_id, instance.name,
                                    'Error', db_status, None, None, 'error')
            raise

        log.info(f'schedule job {id} on {instance}: made job config')

        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=2)) as session:
                url = (f'http://{instance.ip_address}:5000'
                       f'/api/v1alpha/batches/jobs/create')
                await session.post(url, json=body)
                await instance.mark_healthy()
        except aiohttp.ClientResponseError as e:
            await instance.mark_healthy()
            if e.status == 403:
                log.info(f'attempt already exists for job {id} on {instance}, aborting')
            raise e
        except Exception:
            await instance.incr_failed_request_count()
            raise

        log.info(f'schedule job {id} on {instance}: called create job')

        rv = await db.execute_and_fetchone(
            '''
CALL schedule_job(%s, %s, %s, %s);
''',
            (batch_id, job_id, attempt_id, instance.name))
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
