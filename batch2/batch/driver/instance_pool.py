import json
import asyncio
import sortedcontainers
import logging
import aiohttp
import googleapiclient.errors
from hailtop.utils import request_retry_transient_errors

from ..utils import new_token
from ..batch_configuration import BATCH_NAMESPACE, BATCH_WORKER_IMAGE, INSTANCE_ID, \
    PROJECT, ZONE, WORKER_TYPE, WORKER_CORES, WORKER_DISK_SIZE_GB, \
    POOL_SIZE, MAX_INSTANCES

from .instance import Instance
from ..database import check_call_procedure

log = logging.getLogger('instance_pool')

WORKER_CORES_MCPU = WORKER_CORES * 1000

log.info(f'WORKER_CORES {WORKER_CORES}')
log.info(f'WORKER_TYPE {WORKER_TYPE}')
log.info(f'WORKER_DISK_SIZE_GB {WORKER_DISK_SIZE_GB}')
log.info(f'POOL_SIZE {POOL_SIZE}')
log.info(f'MAX_INSTANCES {MAX_INSTANCES}')


class InstancePool:
    def __init__(self, scheduler_state_changed, db, gservices, k8s, bucket_name, machine_name_prefix):
        self.scheduler_state_changed = scheduler_state_changed
        self.db = db
        self.gservices = gservices
        self.k8s = k8s
        self.machine_name_prefix = machine_name_prefix

        if WORKER_TYPE == 'standard':
            m = 3.75
        elif WORKER_TYPE == 'highmem':
            m = 6.5
        else:
            assert WORKER_TYPE == 'highcpu', WORKER_TYPE
            m = 0.9
        self.worker_memory = 0.9 * m

        self.worker_logs_directory = f'gs://{bucket_name}/{BATCH_NAMESPACE}/{INSTANCE_ID}'
        log.info(f'writing worker logs to {self.worker_logs_directory}')

        # active instances only
        self.active_instances_by_free_cores = sortedcontainers.SortedSet(key=lambda inst: inst.free_cores_mcpu)

        self.n_instances_by_state = {
            'pending': 0,
            'active': 0,
            'inactive': 0,
            'deleted': 0
        }

        # pending and active
        self.live_free_cores_mcpu = 0

        self.id_instance = {}

        # FIXME remove
        self.token_inst = {}

    async def async_init(self):
        log.info('initializing instance pool')

        # FIXME async generator
        for record in await self.db.instances.get_all_records():
            instance = Instance.from_record(record)
            self.add_instance(instance)

        # FIXME heal loop
        asyncio.ensure_future(self.event_loop())
        asyncio.ensure_future(self.control_loop())

    @property
    def n_instances(self):
        return len(self.token_inst)

    def adjust_for_remove_instance(self, instance):
        self.n_instances_by_state[instance.state] -= 1

        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu -= instance.free_cores_mcpu
        if instance.state == 'active':
            self.active_instances_by_free_cores.remove(instance)

    async def remove_instance(self, instance):
        async with self.db.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('DELETE FROM instances WHERE id = %s;', (instance.id,))

        self.adjust_for_remove_instance(self, instance)

        del self.token_inst[instance.token]
        del self.id_instance[instance.id]

    def adjust_for_add_instance(self, instance):
        self.n_instances_by_state[instance.state] += 1

        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu += instance.free_cores_mcpu
        if instance.state == 'active':
            self.active_instances_by_free_cores.add(instance)

    def add_instance(self, instance):
        assert instance.token not in self.token_inst
        self.token_inst[instance.token] = instance
        self.id_instance[instance.id] = instance

        self.adjust_for_add_instance(instance)

    async def job_config(self, record):
        job_spec = json.loads(record['spec'])

        secrets = job_spec['secrets']

        secret_futures = []
        for secret in secrets:
            # FIXME need access control to verify user is allowed to access secret
            secret_futures.append(self.k8s.read_secret(secret['name']))
            k8s_secrets = await asyncio.gather(*secret_futures)

        for secret, k8s_secret in zip(secrets, k8s_secrets):
            if k8s_secret:
                secret['data'] = k8s_secret.data

        return {
            'batch_id': record['batch_id'],
            'user': record['user'],
            'job_spec': job_spec,
            'output_directory': record['directory']
        }

    async def schedule_job(self, record, instance):
        assert instance.state == 'active'

        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            url = f'http://{instance.ip_address}:5000/api/v1alpha/batches/jobs/create'
            await request_retry_transient_errors(
                session, 'POST',
                url, json=await self.job_config(record))

        log.info(f'({record["batch_id"]}, {record["job_id"]}) on {instance}: called create job')

        await check_call_procedure(
            self.db.pool,
            'CALL schedule_job(%s, %s, %s);',
            (record['batch_id'], record['job_id'], instance.id))

        log.info(f'({record["batch_id"]}, {record["job_id"]}) on {instance}: updated database')

        self.adjust_for_remove_instance(instance)
        instance.free_cores_mcpu -= record['cores_mcpu']
        self.adjust_for_add_instance(instance)

        log.info(f'({record["batch_id"]}, {record["job_id"]}) on {instance}: adjusted instance pool')

    async def create_instance(self):
        while True:
            inst_token = new_token()
            if inst_token not in self.token_inst:
                break
        machine_name = f'{self.machine_name_prefix}{inst_token}'

        state = 'pending'
        id = await self.db.instances.new_record(
            state=state, name=machine_name, token=inst_token,
            cores_mcpu=WORKER_CORES_MCPU, free_cores_mcpu=WORKER_CORES_MCPU)
        instance = Instance(id, state, machine_name, inst_token, WORKER_CORES_MCPU, WORKER_CORES_MCPU, None)
        self.add_instance(instance)

        log.info(f'created instance {instance}')

        config = {
            'name': machine_name,
            'machineType': f'projects/{PROJECT}/zones/{ZONE}/machineTypes/n1-{WORKER_TYPE}-{WORKER_CORES}',
            'labels': {
                'role': 'batch2-agent',
                'inst_token': inst_token,
                'batch_instance': INSTANCE_ID,
                'namespace': BATCH_NAMESPACE
            },

            'disks': [{
                'boot': True,
                'autoDelete': True,
                'diskSizeGb': WORKER_DISK_SIZE_GB,
                'initializeParams': {
                    'sourceImage': f'projects/{PROJECT}/global/images/batch2-worker-5',
                }
            }],

            'networkInterfaces': [{
                'network': 'global/networks/default',
                'networkTier': 'PREMIUM',
                'accessConfigs': [{
                    'type': 'ONE_TO_ONE_NAT',
                    'name': 'external-nat'
                }]
            }],

            'scheduling': {
                'automaticRestart': False,
                'onHostMaintenance': "TERMINATE",
                'preemptible': True
            },

            'serviceAccounts': [{
                'email': 'batch2-agent@hail-vdc.iam.gserviceaccount.com',
                'scopes': [
                    'https://www.googleapis.com/auth/cloud-platform'
                ]
            }],

            'metadata': {
                'items': [{
                    'key': 'startup-script',
                    'value': f'''
#!/bin/bash
set -ex

export BATCH_WORKER_IMAGE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_worker_image")
export HOME=/root

function retry {{
    local n=1
    local max=5
    local delay=15
    while true; do
        "$@" > worker.log 2>&1 && break || {{
            if [[ $n -lt $max ]]; then
                ((n++))
                echo "Command failed. Attempt $n/$max:"
                sleep $delay;
            else
                export INST_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/inst_token")
                export WORKER_LOGS_DIRECTORY=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker_logs_directory")

                echo "startup of batch worker failed after $n attempts;" >> worker.log
                gsutil -m cp worker.log $WORKER_LOGS_DIRECTORY/$INST_TOKEN/

                export NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
                export ZONE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')
                gcloud -q compute instances delete $NAME --zone=$ZONE
             fi
        }}
    done
}}

retry docker run \
           -v /var/run/docker.sock:/var/run/docker.sock \
           -v /usr/bin/docker:/usr/bin/docker \
           -v /batch:/batch \
           -p 5000:5000 \
           -d --entrypoint "/bin/bash" \
           $BATCH_WORKER_IMAGE \
           -c "sh /run-worker.sh"
'''
                }, {
                    'key': 'inst_token',
                    'value': inst_token
                }, {
                    'key': 'batch_worker_image',
                    'value': BATCH_WORKER_IMAGE
                }, {
                    'key': 'batch_instance',
                    'value': INSTANCE_ID
                }, {
                    'key': 'namespace',
                    'value': BATCH_NAMESPACE
                }, {
                    'key': 'worker_logs_directory',
                    'value': self.worker_logs_directory
                }]
            },
            'tags': {
                'items': [
                    "batch2-agent"
                ]
            },
        }

        await self.gservices.create_instance(config)
        log.info(f'created machine {machine_name} with logs at {self.worker_logs_directory}/{inst_token}/worker.log')

    async def activate_instance(self, instance, ip_address):
        assert instance.state == 'pending'

        check_call_procedure(
            self.db.pool,
            'CALL activate_instance(%s, %s);',
            (instance.id, ip_address))

        self.adjust_for_remove_instance(instance)
        instance.state = 'active'
        instance.ip_address = ip_address
        self.adjust_for_add_instance(instance)

        self.scheduler_state_changed.set()

    async def deactivate_instance(self, instance):
        if instance.state in ('inactive', 'deleted'):
            return

        check_call_procedure(
            self.db.pool,
            'CALL deactivate_instance(%s);',
            (instance.id,))

        self.adjust_for_remove_instance(instance)
        instance.state = 'inactive'
        instance.free_cores_mcpu = instance.cores_mcpu
        self.adjust_for_add_instance(instance)
        # there might be jobs to reschedule
        self.scheduler_state_changed.set()

    async def call_delete_instance(self, instance):
        if instance.state == 'deleted':
            return
        assert instance.state == 'inactive'

        try:
            await self.gservices.delete_instance(instance.name)
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                log.info(f'{instance} already delete done')
                await self.remove_instance(instance)
                return
            raise

        check_call_procedure(
            self.db.pool,
            'CALL mark_instance_deleted(%s);',
            (instance.id,))

        self.adjust_for_remove_instance(instance)
        instance.state = 'deleted'
        self.adjust_for_add_instance(instance)

    async def handle_preempt_event(self, instance):
        await self.deactivate_instance(instance)
        await self.call_delete_instance(instance)

    async def handle_delete_done_event(self, instance):
        await self.remove_instance(instance)

    async def handle_call_delete_event(self, instance):
        await self.deactivate_instance(instance)

    async def handle_event(self, event):
        if not event.payload:
            log.warning(f'event has no payload')
            return

        payload = event.payload
        version = payload['version']
        if version != '1.2':
            log.warning('unknown event verison {version}')
            return

        resource_type = event.resource.type
        if resource_type != 'gce_instance':
            log.warning(f'unknown event resource type {resource_type}')
            return

        event_type = payload['event_type']
        event_subtype = payload['event_subtype']
        resource = payload['resource']
        name = resource['name']

        log.info(f'event {version} {resource_type} {event_type} {event_subtype} {name}')

        if not name.startswith(self.machine_name_prefix):
            log.warning(f'event for unknown machine {name}')
            return

        inst_token = name[len(self.machine_name_prefix):]
        instance = self.token_inst.get(inst_token)
        if not instance:
            log.warning(f'event for unknown instance {inst_token}')
            return

        if event_subtype == 'compute.instances.preempted':
            log.info(f'event handler: handle preempt {instance}')
            await self.handle_preempt_event(instance)
        elif event_subtype == 'compute.instances.delete':
            if event_type == 'GCE_OPERATION_DONE':
                log.info(f'event handler: delete {instance} done')
                await self.handle_delete_done_event(instance)
            elif event_type == 'GCE_API_CALL':
                log.info(f'event handler: handle call delete {instance}')
                await self.handle_call_delete_event(instance)
            else:
                log.warning(f'unknown event type {event_type}')
        else:
            log.warning(f'unknown event subtype {event_subtype}')

    async def event_loop(self):
        log.info(f'starting event loop')
        while True:
            try:
                async for event in await self.gservices.stream_entries():
                    await self.handle_event(event)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception('event loop failed due to exception')
            await asyncio.sleep(15)

    async def control_loop(self):
        log.info(f'starting control loop')
        while True:
            try:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute('SELECT * FROM ready_cores')
                        row = await cursor.fetchone()
                        ready_cores_mcpu = row['ready_cores_mcpu']

                log.info(f'n_instances {self.n_instances_by_state}'
                         f' n_instances {self.n_instances}'
                         f' live_free_cores {self.live_free_cores_mcpu / 1000}'
                         f' ready_cores {ready_cores_mcpu / 1000}')

                if ready_cores_mcpu > 0:
                    n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
                    instances_needed = (ready_cores_mcpu - self.live_free_cores_mcpu + WORKER_CORES_MCPU - 1) // WORKER_CORES_MCPU
                    instances_needed = min(instances_needed,
                                           POOL_SIZE - n_live_instances,
                                           MAX_INSTANCES - self.n_instances,
                                           # 20 queries/s; our GCE long-run quota
                                           300)
                    if instances_needed > 0:
                        log.info(f'creating {instances_needed} new instances')
                        # parallelism will be bounded by thread pool
                        await asyncio.gather(*[self.create_instance() for _ in range(instances_needed)])
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception('instance pool control loop: caught exception')

            await asyncio.sleep(15)
