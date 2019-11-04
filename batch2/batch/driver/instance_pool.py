import time
import secrets
import asyncio
import logging
import sortedcontainers
import aiohttp
import googleapiclient.errors

from ..batch_configuration import BATCH_NAMESPACE, BATCH_WORKER_IMAGE, \
    PROJECT, ZONE, WORKER_TYPE, WORKER_CORES, WORKER_DISK_SIZE_GB, \
    POOL_SIZE, MAX_INSTANCES

from .instance import Instance

log = logging.getLogger('instance_pool')

WORKER_CORES_MCPU = WORKER_CORES * 1000

log.info(f'WORKER_CORES {WORKER_CORES}')
log.info(f'WORKER_TYPE {WORKER_TYPE}')
log.info(f'WORKER_DISK_SIZE_GB {WORKER_DISK_SIZE_GB}')
log.info(f'POOL_SIZE {POOL_SIZE}')
log.info(f'MAX_INSTANCES {MAX_INSTANCES}')


class InstancePool:
    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.instance_id = app['instance_id']
        self.log_store = app['log_store']
        self.scheduler_state_changed = app['scheduler_state_changed']
        self.db = app['db']
        self.gservices = app['gservices']
        self.k8s = app['k8s_client']
        self.machine_name_prefix = machine_name_prefix

        if WORKER_TYPE == 'standard':
            m = 3.75
        elif WORKER_TYPE == 'highmem':
            m = 6.5
        else:
            assert WORKER_TYPE == 'highcpu', WORKER_TYPE
            m = 0.9
        self.worker_memory = 0.9 * m

        self.instances_by_last_updated = sortedcontainers.SortedSet(
            key=lambda instance: instance.last_updated)

        self.healthy_instances_by_free_cores = sortedcontainers.SortedSet(
            key=lambda instance: instance.free_cores_mcpu)

        self.n_instances_by_state = {
            'pending': 0,
            'active': 0,
            'inactive': 0,
            'deleted': 0
        }

        # pending and active
        self.live_free_cores_mcpu = 0

        self.name_instance = {}

    async def async_init(self):
        log.info('initializing instance pool')

        async for record in self.db.execute_and_fetchall(
                'SELECT * FROM instances;'):
            instance = Instance.from_record(self.app, record)
            self.add_instance(instance)

        asyncio.ensure_future(self.event_loop())
        asyncio.ensure_future(self.control_loop())
        asyncio.ensure_future(self.instance_monitoring_loop())

    @property
    def n_instances(self):
        return len(self.name_instance)

    def adjust_for_remove_instance(self, instance):
        self.n_instances_by_state[instance.state] -= 1

        self.instances_by_last_updated.remove(instance)
        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu -= instance.free_cores_mcpu
        if instance in self.healthy_instances_by_free_cores:
            self.healthy_instances_by_free_cores.remove(instance)

    async def remove_instance(self, instance):
        await instance.deactivate()

        await self.db.just_execute(
            'DELETE FROM instances WHERE name = %s;', (instance.name,))

        self.adjust_for_remove_instance(instance)

        del self.name_instance[instance.name]

    def adjust_for_add_instance(self, instance):
        self.n_instances_by_state[instance.state] += 1

        self.instances_by_last_updated.add(instance)
        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu += instance.free_cores_mcpu
        if (instance.state == 'active' and
                instance.failed_request_count <= 1):
            self.healthy_instances_by_free_cores.add(instance)

    def add_instance(self, instance):
        assert instance.name not in self.name_instance
        self.name_instance[instance.name] = instance

        self.adjust_for_add_instance(instance)

    async def create_instance(self):
        while True:
            # 36 ** 5 = ~60M
            suffix = ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789')
                              for _ in range(5)])
            machine_name = f'{self.machine_name_prefix}{suffix}'
            if machine_name not in self.name_instance:
                break

        activation_token = secrets.token_urlsafe(32)
        instance = await Instance.create(self.app, machine_name, activation_token, WORKER_CORES_MCPU)
        self.add_instance(instance)

        log.info(f'created {instance}')

        config = {
            'name': machine_name,
            'machineType': f'projects/{PROJECT}/zones/{ZONE}/machineTypes/n1-{WORKER_TYPE}-{WORKER_CORES}',
            'labels': {
                'role': 'batch2-agent',
                'namespace': BATCH_NAMESPACE
            },

            'disks': [{
                'boot': True,
                'autoDelete': True,
                'diskSizeGb': WORKER_DISK_SIZE_GB,
                'initializeParams': {
                    'sourceImage': f'projects/{PROJECT}/global/images/batch2-worker-6',
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
set -x

curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_script"  >./run.sh

nohup /bin/bash run.sh >run.log 2>&1 &
'''
                }, {
                    'key': 'run_script',
                    'value': '''
#!/bin/bash
set -x

# only allow udp/53 (dns) to metadata server
# -I inserts at the head of the chain, so the ACCEPT rule runs first
iptables -I FORWARD -i docker0 -d 169.254.169.254 -j DROP
iptables -I FORWARD -i docker0 -d 169.254.169.254 -p udp -m udp --destination-port 53 -j ACCEPT

export HOME=/root

CORES=$(nproc)
NAMESPACE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/namespace")
ACTIVATION_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/activation_token")
IP_ADDRESS=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip")
PROJECT=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/project/project-id")

BUCKET_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket_name")
INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_id")
NAME=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
ZONE=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')

BATCH_WORKER_IMAGE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_worker_image")

# retry once
docker pull $BATCH_WORKER_IMAGE || \
    (echo 'pull failed, retrying' && sleep 15 && docker pull $BATCH_WORKER_IMAGE)

# So here I go it's my shot.
docker run \
    -e CORES=$CORES \
    -e NAME=$NAME \
    -e NAMESPACE=$NAMESPACE \
    -e ACTIVATION_TOKEN=$ACTIVATION_TOKEN \
    -e IP_ADDRESS=$IP_ADDRESS \
    -e BUCKET_NAME=$BUCKET_NAME \
    -e INSTANCE_ID=$INSTANCE_ID \
    -e PROJECT=$PROJECT \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /batch:/batch \
    -v /logs:/logs \
    -p 5000:5000 \
    $BATCH_WORKER_IMAGE \
    python3 -u -m batch.worker >worker.log 2>&1

# this has to match LogStore.worker_log_path
gsutil -m cp run.log worker.log /var/log/syslog gs://$BUCKET_NAME/batch2/logs/$INSTANCE_ID/worker/$NAME/

while true; do
  gcloud -q compute instances delete $NAME --zone=$ZONE
  sleep 1
done
'''
                }, {
                    'key': 'activation_token',
                    'value': activation_token
                }, {
                    'key': 'batch_worker_image',
                    'value': BATCH_WORKER_IMAGE
                }, {
                    'key': 'namespace',
                    'value': BATCH_NAMESPACE
                }, {
                    'key': 'bucket_name',
                    'value': self.log_store.bucket_name
                }, {
                    'key': 'instance_id',
                    'value': self.log_store.instance_id
                }]
            },
            'tags': {
                'items': [
                    "batch2-agent"
                ]
            },
        }

        await self.gservices.create_instance(config)

        log.info(f'created machine {machine_name} for {instance} '
                 f' with logs at {self.log_store.worker_log_path(machine_name, "worker.log")}')

    async def call_delete_instance(self, instance, force=False):
        if instance.state == 'deleted' and not force:
            return
        if instance.state not in ('inactive', 'deleted'):
            await instance.deactivate()

        try:
            await self.gservices.delete_instance(instance.name)
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                log.info(f'{instance} already delete done')
                await self.remove_instance(instance)
                return
            raise

    async def handle_preempt_event(self, instance):
        await self.call_delete_instance(instance)

    async def handle_delete_done_event(self, instance):
        await self.remove_instance(instance)

    async def handle_call_delete_event(self, instance):
        await instance.mark_deleted()

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

        instance = self.name_instance.get(name)
        if not instance:
            log.warning(f'event for unknown instance {name}')
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
                log.exception('in event loop')
            await asyncio.sleep(15)

    async def control_loop(self):
        log.info(f'starting control loop')
        while True:
            try:
                ready_cores = await self.db.execute_and_fetchone(
                    'SELECT * FROM ready_cores;')
                ready_cores_mcpu = ready_cores['ready_cores_mcpu']

                log.info(f'n_instances {self.n_instances} {self.n_instances_by_state}'
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
                log.exception('in control loop')
            await asyncio.sleep(15)

    async def check_on_instance(self, instance):
        if instance.ip_address:
            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                    await session.get(f'http://{instance.ip_address}:5000/healthcheck')
                    await instance.mark_healthy()
                    return
            except Exception:
                log.exception(f'while requesting {instance} /healthcheck')
                await instance.incr_failed_request_count()

        try:
            spec = await self.gservices.get_instance(instance.name)
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                await self.remove_instance(instance)
                return

        # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED
        gce_state = spec['status']
        log.info(f'{instance} gce_state {gce_state}')

        if gce_state in ('STOPPING', 'TERMINATED'):
            log.info(f'{instance} live but stopping or terminated, deactivating')
            await instance.deactivate()

        if (gce_state in ('STAGING', 'RUNNING') and
                instance.state == 'pending' and
                time.time() - instance.time_created > 5 * 60):
            # FIXME shouldn't count time in PROVISIONING
            log.info(f'{instance} did not activate within 5m, deleting')
            await self.call_delete_instance(instance)

        if instance.state == 'inactive':
            log.info(f'{instance} is inactive, deleting')
            await self.call_delete_instance(instance)

        await instance.update_timestamp()

    async def instance_monitoring_loop(self):
        log.info(f'starting instance monitoring loop')

        while True:
            try:
                if self.instances_by_last_updated:
                    # 0 is the smallest (oldest)
                    instance = self.instances_by_last_updated[0]
                    since_last_updated = time.time() - instance.last_updated
                    if since_last_updated > 60:
                        log.info(f'checking on {instance}, last updated {since_last_updated}s ago')
                        await self.check_on_instance(instance)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception('in monitor instances loop')

            await asyncio.sleep(1)
