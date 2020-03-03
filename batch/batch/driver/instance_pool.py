import secrets
import random
import asyncio
import logging
import sortedcontainers
import googleapiclient.errors
from hailtop.utils import time_msecs

from ..batch_configuration import DEFAULT_NAMESPACE, BATCH_WORKER_IMAGE, \
    PROJECT

from .instance import Instance

log = logging.getLogger('instance_pool')


class InstancePool:
    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.instance_id = app['instance_id']
        self.log_store = app['log_store']
        self.scheduler_state_changed = app['scheduler_state_changed']
        self.db = app['db']
        self.gservices = app['gservices']
        self.machine_name_prefix = machine_name_prefix

        # set in async_init
        self.worker_type = None
        self.worker_cores = None
        self.worker_disk_size_gb = None
        self.max_instances = None
        self.pool_size = None

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

        row = await self.db.select_and_fetchone(
            'SELECT worker_type, worker_cores, worker_disk_size_gb, max_instances, pool_size FROM globals;')

        self.worker_type = row['worker_type']
        self.worker_cores = row['worker_cores']
        self.worker_disk_size_gb = row['worker_disk_size_gb']
        self.max_instances = row['max_instances']
        self.pool_size = row['pool_size']

        async for record in self.db.select_and_fetchall(
                'SELECT * FROM instances WHERE removed = 0;'):
            instance = Instance.from_record(self.app, record)
            self.add_instance(instance)

        asyncio.ensure_future(self.event_loop())
        asyncio.ensure_future(self.control_loop())
        asyncio.ensure_future(self.instance_monitoring_loop())

    def config(self):
        return {
            'worker_type': self.worker_type,
            'worker_cores': self.worker_cores,
            'worker_disk_size_gb': self.worker_disk_size_gb,
            'max_instances': self.max_instances,
            'pool_size': self.pool_size
        }

    # FIXME can't adjust worker type, cores because we check if jobs
    # can be scheduled in the front-end before inserting into the
    # database
    async def configure(
            self,
            # worker_type, worker_cores, worker_disk_size_gb,
            max_instances, pool_size):
        await self.db.just_execute(
            # worker_type = %s, worker_cores = %s, worker_disk_size_gb = %s,
            '''
UPDATE globals
SET max_instances = %s, pool_size = %s;
''',
            (
                # worker_type, worker_cores, worker_disk_size_gb,
                max_instances, pool_size))
        # self.worker_type = worker_type
        # self.worker_cores = worker_cores
        # self.worker_disk_size_gb = worker_disk_size_gb
        self.max_instances = max_instances
        self.pool_size = pool_size

    @property
    def n_instances(self):
        return len(self.name_instance)

    def adjust_for_remove_instance(self, instance):
        assert instance in self.instances_by_last_updated

        self.instances_by_last_updated.remove(instance)

        self.n_instances_by_state[instance.state] -= 1

        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu -= max(0, instance.free_cores_mcpu)
        if instance in self.healthy_instances_by_free_cores:
            self.healthy_instances_by_free_cores.remove(instance)

    async def remove_instance(self, instance, reason, timestamp=None):
        await instance.deactivate(reason, timestamp)

        await self.db.just_execute(
            'UPDATE instances SET removed = 1 WHERE name = %s;', (instance.name,))

        self.adjust_for_remove_instance(instance)
        del self.name_instance[instance.name]

    def adjust_for_add_instance(self, instance):
        assert instance not in self.instances_by_last_updated

        self.n_instances_by_state[instance.state] += 1

        self.instances_by_last_updated.add(instance)
        if instance.state in ('pending', 'active'):
            self.live_free_cores_mcpu += max(0, instance.free_cores_mcpu)
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

        n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
        total_cores = self.worker_cores * n_live_instances

        if total_cores < 5_000:
            zones = ['us-central1-a', 'us-central1-b', 'us-central1-c', 'us-central1-f']
            zone = random.choice(zones)
        else:
            zones = ['us-central1-a', 'us-central1-b', 'us-central1-c', 'us-central1-f', 'us-east1-a', 'us-east1-b', 'us-east1-c', 'us-east4-a', 'us-east4-b', 'us-east4-c', 'us-west1-a', 'us-west1-b', 'us-west1-c', 'us-west2-a', 'us-west2-b', 'us-west2-c']
            # based on quotas, us-central1: 300K over 4 zones, rest: 100K over 3 zones
            weights = 4 * [295 / 4] + 12 * [100 / 3]

            zone = random.choices(zones, weights)[0]

        activation_token = secrets.token_urlsafe(32)
        instance = await Instance.create(self.app, machine_name, activation_token, self.worker_cores * 1000, zone)
        self.add_instance(instance)

        log.info(f'created {instance}')

        config = {
            'name': machine_name,
            'machineType': f'projects/{PROJECT}/zones/{zone}/machineTypes/n1-{self.worker_type}-{self.worker_cores}',
            'labels': {
                'role': 'batch2-agent',
                'namespace': DEFAULT_NAMESPACE
            },

            'disks': [{
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': f'projects/{PROJECT}/global/images/batch-worker-7',
                    'diskType': f'projects/{PROJECT}/zones/{zone}/diskTypes/pd-ssd',
                    'diskSizeGb': str(self.worker_disk_size_gb)
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
                'email': f'batch2-agent@{PROJECT}.iam.gserviceaccount.com',
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
WORKER_TYPE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker_type")
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
    -e WORKER_TYPE=$WORKER_TYPE \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /batch:/batch \
    -v /logs:/logs \
    -p 5000:5000 \
    $BATCH_WORKER_IMAGE \
    python3 -u -m batch.worker >worker.log 2>&1

while true; do
  gcloud -q compute instances delete $NAME --zone=$ZONE
  sleep 1
done
'''
                }, {
                    'key': 'shutdown-script',
                    'value': '''
set -x

BUCKET_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket_name")
INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_id")
NAME=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')

# this has to match LogStore.worker_log_path
gsutil -m cp run.log worker.log /var/log/syslog gs://$BUCKET_NAME/batch/logs/$INSTANCE_ID/worker/$NAME/
'''
                }, {
                    'key': 'activation_token',
                    'value': activation_token
                }, {
                    'key': 'batch_worker_image',
                    'value': BATCH_WORKER_IMAGE
                }, {
                    'key': 'namespace',
                    'value': DEFAULT_NAMESPACE
                }, {
                    'key': 'bucket_name',
                    'value': self.log_store.bucket_name
                }, {
                    'key': 'instance_id',
                    'value': self.log_store.instance_id
                }, {
                    'key': 'worker_type',
                    'value': self.worker_type
                }]
            },
            'tags': {
                'items': [
                    "batch2-agent"
                ]
            },
        }

        await self.gservices.create_instance(config, zone)

        log.info(f'created machine {machine_name} for {instance} '
                 f' with logs at {self.log_store.worker_log_path(machine_name, "worker.log")}')

    async def call_delete_instance(self, instance, reason, timestamp=None, force=False):
        if instance.state == 'deleted' and not force:
            return
        if instance.state not in ('inactive', 'deleted'):
            await instance.deactivate(reason, timestamp)

        try:
            await self.gservices.delete_instance(instance.name, instance.zone)
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                log.info(f'{instance} already delete done')
                await self.remove_instance(instance, reason, timestamp)
                return
            raise

    async def handle_preempt_event(self, instance, timestamp):
        await self.call_delete_instance(instance, 'preempted', timestamp=timestamp)

    async def handle_delete_done_event(self, instance, timestamp):
        await self.remove_instance(instance, 'deleted', timestamp)

    async def handle_call_delete_event(self, instance, timestamp):
        await instance.mark_deleted('deleted', timestamp)

    async def handle_event(self, event):
        if not event.payload:
            log.warning(f'event has no payload')
            return

        timestamp = event.timestamp.timestamp() * 1000
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
            await self.handle_preempt_event(instance, timestamp)
        elif event_subtype == 'compute.instances.delete':
            if event_type == 'GCE_OPERATION_DONE':
                log.info(f'event handler: delete {instance} done')
                await self.handle_delete_done_event(instance, timestamp)
            elif event_type == 'GCE_API_CALL':
                log.info(f'event handler: handle call delete {instance}')
                await self.handle_call_delete_event(instance, timestamp)
            else:
                log.warning(f'unknown event type {event_type}')
        else:
            log.warning(f'unknown event subtype {event_subtype}')

    async def event_loop(self):
        log.info(f'starting event loop')
        while True:
            try:
                async for event in await self.gservices.stream_entries(self.db):
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
                ready_cores = await self.db.select_and_fetchone(
                    '''
SELECT CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
FROM ready_cores;
''')
                ready_cores_mcpu = ready_cores['ready_cores_mcpu']

                free_cores_mcpu = sum([
                    worker.free_cores_mcpu
                    for worker in self.healthy_instances_by_free_cores
                ])
                free_cores = free_cores_mcpu / 1000

                log.info(f'n_instances {self.n_instances} {self.n_instances_by_state}'
                         f' free_cores {free_cores} live_free_cores {self.live_free_cores_mcpu / 1000}'
                         f' ready_cores {ready_cores_mcpu / 1000}')

                if ready_cores_mcpu > 0 and free_cores < 500:
                    n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
                    instances_needed = (
                        (ready_cores_mcpu - self.live_free_cores_mcpu + (self.worker_cores * 1000) - 1) //
                        (self.worker_cores * 1000))
                    instances_needed = min(instances_needed,
                                           self.pool_size - n_live_instances,
                                           self.max_instances - self.n_instances,
                                           # 20 queries/s; our GCE long-run quota
                                           300,
                                           # n * 16 cores / 15s = excess_scheduling_rate/s = 10/s => n ~= 10
                                           10)
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
        active_and_healthy = await instance.check_is_active_and_healthy()
        if active_and_healthy:
            return

        try:
            spec = await self.gservices.get_instance(instance.name, instance.zone)
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                await self.remove_instance(instance, 'does_not_exist')
                return
            raise

        # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED
        gce_state = spec['status']
        log.info(f'{instance} gce_state {gce_state}')

        if gce_state in ('STOPPING', 'TERMINATED'):
            log.info(f'{instance} live but stopping or terminated, deactivating')
            await instance.deactivate('terminated')

        if (gce_state in ('STAGING', 'RUNNING') and
                instance.state == 'pending' and
                time_msecs() - instance.time_created > 5 * 60 * 1000):
            # FIXME shouldn't count time in PROVISIONING
            log.info(f'{instance} did not activate within 5m, deleting')
            await self.call_delete_instance(instance, 'activation_timeout')

        if instance.state == 'inactive':
            log.info(f'{instance} is inactive, deleting')
            await self.call_delete_instance(instance, 'inactive')

        await instance.update_timestamp()

    async def instance_monitoring_loop(self):
        log.info(f'starting instance monitoring loop')

        while True:
            try:
                if self.instances_by_last_updated:
                    # 0 is the smallest (oldest)
                    instance = self.instances_by_last_updated[0]
                    since_last_updated = time_msecs() - instance.last_updated
                    if since_last_updated > 60 * 1000:
                        log.info(f'checking on {instance}, last updated {since_last_updated / 1000}s ago')
                        await self.check_on_instance(instance)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception('in monitor instances loop')

            await asyncio.sleep(1)
