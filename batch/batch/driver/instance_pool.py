import os
import secrets
import random
import json
import datetime
import asyncio
import urllib.parse
import logging
import base64
import dateutil.parser
import sortedcontainers
import aiohttp
from hailtop.utils import (
    retry_long_running, time_msecs, secret_alnum_string)

from ..batch_configuration import DEFAULT_NAMESPACE, PROJECT, \
    WORKER_MAX_IDLE_TIME_MSECS, STANDING_WORKER_MAX_IDLE_TIME_MSECS, \
    ENABLE_STANDING_WORKER

from .instance import Instance
from ..worker_config import WorkerConfig

log = logging.getLogger('instance_pool')

BATCH_WORKER_IMAGE = os.environ['HAIL_BATCH_WORKER_IMAGE']

log.info(f'BATCH_WORKER_IMAGE {BATCH_WORKER_IMAGE}')


class InstancePool:
    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.instance_id = app['instance_id']
        self.log_store = app['log_store']
        self.scheduler_state_changed = app['scheduler_state_changed']
        self.db = app['db']
        self.compute_client = app['compute_client']
        self.logging_client = app['logging_client']
        self.machine_name_prefix = machine_name_prefix

        # set in async_init
        self.worker_type = None
        self.worker_cores = None
        self.worker_disk_size_gb = None
        self.worker_local_ssd_data_disk = None
        self.worker_pd_ssd_data_disk_size_gb = None
        self.standing_worker_cores = None
        self.max_instances = None
        self.pool_size = None

        # default until we update zones
        # /regions is slow, don't make it synchronous on startup
        self.zones = ['us-central1-a', 'us-central1-b', 'us-central1-c', 'us-central1-f']
        self.zone_weights = [1, 1, 1, 1]

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
        self.live_total_cores_mcpu = 0

        self.name_instance = {}

    async def update_zones(self):
        northamerica_regions = {
            # 'northamerica-northeast1',
            'us-central1',
            'us-east1',
            'us-east4',
            'us-west1',
            'us-west2',
            'us-west3',
            'us-west4'
        }

        new_zones = []
        new_zone_weights = []

        async for r in await self.compute_client.list('/regions'):
            name = r['name']
            if name not in northamerica_regions:
                continue

            quota_remaining = {
                q['metric']: q['limit'] - q['usage']
                for q in r['quotas']
            }

            remaining = quota_remaining['PREEMPTIBLE_CPUS'] // self.worker_cores
            if self.worker_local_ssd_data_disk:
                remaining = min(remaining, quota_remaining['LOCAL_SSD_TOTAL_GB'] // 375)
            else:
                remaining = min(remaining, quota_remaining['SSD_TOTAL_GB'] // self.worker_pd_ssd_data_disk_size_gb)

            weight = max(remaining // len(r['zones']), 1)
            for z in r['zones']:
                zone_name = os.path.basename(urllib.parse.urlparse(z).path)
                new_zones.append(zone_name)
                new_zone_weights.append(weight)

        self.zones = new_zones
        self.zone_weights = new_zone_weights

        log.info(f'updated zones: zones {self.zones} zone_weights {self.zone_weights}')

    async def update_zones_loop(self):
        while True:
            log.info('update zones loop')
            await self.update_zones()
            await asyncio.sleep(60)

    async def async_init(self):
        log.info('initializing instance pool')

        row = await self.db.select_and_fetchone('''
SELECT worker_type, worker_cores, worker_disk_size_gb,
  worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb,
  standing_worker_cores, max_instances, pool_size
FROM globals;
''')

        self.worker_type = row['worker_type']
        self.worker_cores = row['worker_cores']
        self.worker_disk_size_gb = row['worker_disk_size_gb']
        self.worker_local_ssd_data_disk = row['worker_local_ssd_data_disk']
        self.worker_pd_ssd_data_disk_size_gb = row['worker_pd_ssd_data_disk_size_gb']
        self.standing_worker_cores = row['standing_worker_cores']
        self.max_instances = row['max_instances']
        self.pool_size = row['pool_size']

        async for record in self.db.select_and_fetchall(
                'SELECT * FROM instances WHERE removed = 0;'):
            instance = Instance.from_record(self.app, record)
            self.add_instance(instance)

        asyncio.ensure_future(self.event_loop())
        asyncio.ensure_future(self.control_loop())
        asyncio.ensure_future(self.instance_monitoring_loop())

        asyncio.ensure_future(retry_long_running(
            'update_zones_loop',
            self.update_zones_loop))

    def config(self):
        return {
            'worker_type': self.worker_type,
            'worker_cores': self.worker_cores,
            'worker_disk_size_gb': self.worker_disk_size_gb,
            'worker_local_ssd_data_disk': self.worker_local_ssd_data_disk,
            'worker_pd_ssd_data_disk_size_gb': self.worker_pd_ssd_data_disk_size_gb,
            'standing_worker_cores': self.standing_worker_cores,
            'max_instances': self.max_instances,
            'pool_size': self.pool_size
        }

    async def configure(
            self,
            worker_type, worker_cores, worker_disk_size_gb,
            worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb,
            standing_worker_cores,
            max_instances, pool_size):
        await self.db.just_execute(
            '''
UPDATE globals
SET worker_type = %s, worker_cores = %s, worker_disk_size_gb = %s,
  worker_local_ssd_data_disk = %s, worker_pd_ssd_data_disk_size_gb = %s,
  standing_worker_cores = %s, max_instances = %s, pool_size = %s;
''',
            (worker_type, worker_cores, worker_disk_size_gb,
             worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb,
             standing_worker_cores,
             max_instances, pool_size))
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        self.worker_disk_size_gb = worker_disk_size_gb
        self.worker_local_ssd_data_disk = worker_local_ssd_data_disk
        self.worker_pd_ssd_data_disk_size_gb = worker_pd_ssd_data_disk_size_gb
        self.standing_worker_cores = standing_worker_cores
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
            self.live_total_cores_mcpu -= instance.cores_mcpu
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
            self.live_total_cores_mcpu += instance.cores_mcpu
        if (instance.state == 'active'
                and instance.failed_request_count <= 1):
            self.healthy_instances_by_free_cores.add(instance)

    def add_instance(self, instance):
        assert instance.name not in self.name_instance

        self.name_instance[instance.name] = instance
        self.adjust_for_add_instance(instance)

    async def create_instance(self, cores=None, max_idle_time_msecs=None):
        if cores is None:
            cores = self.worker_cores
        if max_idle_time_msecs is None:
            max_idle_time_msecs = WORKER_MAX_IDLE_TIME_MSECS

        while True:
            # 36 ** 5 = ~60M
            suffix = secret_alnum_string(5, case='lower')
            machine_name = f'{self.machine_name_prefix}{suffix}'
            if machine_name not in self.name_instance:
                break

        if self.live_total_cores_mcpu // 1000 < 4_000:
            zones = ['us-central1-a', 'us-central1-b', 'us-central1-c', 'us-central1-f']
            zone = random.choice(zones)
        else:
            zone = random.choices(self.zones, self.zone_weights)[0]

        activation_token = secrets.token_urlsafe(32)
        instance = await Instance.create(self.app, machine_name, activation_token, cores * 1000, zone)
        self.add_instance(instance)

        log.info(f'created {instance}')

        if self.worker_local_ssd_data_disk:
            worker_data_disk = {
                'type': 'SCRATCH',
                'autoDelete': True,
                'interface': 'NVME',
                'initializeParams': {
                    'diskType': f'zones/{zone}/diskTypes/local-ssd'
                }}
            worker_data_disk_name = 'nvme0n1'
        else:
            worker_data_disk = {
                'autoDelete': True,
                'initializeParams': {
                    'diskType': f'projects/{PROJECT}/zones/{zone}/diskTypes/pd-ssd',
                    'diskSizeGb': str(self.worker_pd_ssd_data_disk_size_gb)
                }
            }
            worker_data_disk_name = 'sdb'

        config = {
            'name': machine_name,
            'machineType': f'projects/{PROJECT}/zones/{zone}/machineTypes/n1-{self.worker_type}-{cores}',
            'labels': {
                'role': 'batch2-agent',
                'namespace': DEFAULT_NAMESPACE
            },

            'disks': [{
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': f'projects/{PROJECT}/global/images/batch-worker-9',
                    'diskType': f'projects/{PROJECT}/zones/{zone}/diskTypes/pd-ssd',
                    'diskSizeGb': str(self.worker_disk_size_gb)
                }
            }, worker_data_disk],

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
                    'value': '''
#!/bin/bash
set -x

curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_script"  >./run.sh

nohup /bin/bash run.sh >run.log 2>&1 &
'''
                }, {
                    'key': 'run_script',
                    'value': rf'''
#!/bin/bash
set -x

# only allow udp/53 (dns) to metadata server
# -I inserts at the head of the chain, so the ACCEPT rule runs first
iptables -I FORWARD -i docker0 -d 169.254.169.254 -j DROP
iptables -I FORWARD -i docker0 -d 169.254.169.254 -p udp -m udp --destination-port 53 -j ACCEPT

# add docker daemon debug logging
jq '.debug = true' /etc/docker/daemon.json > daemon.json.tmp
mv daemon.json.tmp /etc/docker/daemon.json
kill -SIGHUP $(pidof dockerd)

WORKER_DATA_DISK_NAME="{worker_data_disk_name}"

# format local SSD
sudo mkfs.xfs -m reflink=1 /dev/$WORKER_DATA_DISK_NAME
sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME
sudo mount -o prjquota /dev/$WORKER_DATA_DISK_NAME /mnt/disks/$WORKER_DATA_DISK_NAME
sudo chmod a+w /mnt/disks/$WORKER_DATA_DISK_NAME
XFS_DEVICE=$(xfs_info /mnt/disks/$WORKER_DATA_DISK_NAME | head -n 1 | awk '{{ print $1 }}' | awk  'BEGIN {{ FS = "=" }}; {{ print $2 }}')

# reconfigure docker to use local SSD
sudo service docker stop
sudo mv /var/lib/docker /mnt/disks/$WORKER_DATA_DISK_NAME/docker
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/docker /var/lib/docker
sudo service docker start

# reconfigure /batch and /logs and /gcsfuse to use local SSD
sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/batch/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/batch /batch

sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/logs/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/logs /logs

sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/gcsfuse/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/gcsfuse /gcsfuse

sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/xfsquota/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/xfsquota /xfsquota

touch /xfsquota/projects
touch /xfsquota/projid

ln -s /xfsquota/projects /etc/projects
ln -s /xfsquota/projid /etc/projid

export HOME=/root

CORES=$(nproc)
NAMESPACE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/namespace")
ACTIVATION_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/activation_token")
IP_ADDRESS=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip")
PROJECT=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/project/project-id")

BATCH_LOGS_BUCKET_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_logs_bucket_name")
WORKER_LOGS_BUCKET_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker_logs_bucket_name")
INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_id")
WORKER_CONFIG=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker_config")
MAX_IDLE_TIME_MSECS=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/max_idle_time_msecs")
NAME=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
ZONE=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')

BATCH_WORKER_IMAGE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_worker_image")

# Setup fluentd
touch /worker.log
touch /run.log

sudo rm /etc/google-fluentd/config.d/*  # remove unused config files

sudo tee /etc/google-fluentd/config.d/syslog.conf <<EOF
<source>
  @type tail
  format syslog
  path /var/log/syslog
  pos_file /var/lib/google-fluentd/pos/syslog.pos
  read_from_head true
  tag syslog
</source>
EOF

sudo tee /etc/google-fluentd/config.d/worker-log.conf <<EOF {{
<source>
    @type tail
    format json
    path /worker.log
    pos_file /var/lib/google-fluentd/pos/worker-log.pos
    read_from_head true
    tag worker.log
</source>

<filter worker.log>
    @type record_transformer
    enable_ruby
    <record>
        severity \${{ record["levelname"] }}
        timestamp \${{ record["asctime"] }}
    </record>
</filter>
EOF

sudo tee /etc/google-fluentd/config.d/run-log.conf <<EOF
<source>
    @type tail
    format none
    path /run.log
    pos_file /var/lib/google-fluentd/pos/run-log.pos
    read_from_head true
    tag run.log
</source>
EOF

sudo cp /etc/google-fluentd/google-fluentd.conf /etc/google-fluentd/google-fluentd.conf.bak
head -n -1 /etc/google-fluentd/google-fluentd.conf.bak | sudo tee /etc/google-fluentd/google-fluentd.conf
sudo tee -a /etc/google-fluentd/google-fluentd.conf <<EOF
  labels {{
    "namespace": "$NAMESPACE",
    "instance_id": "$INSTANCE_ID"
  }}
</match>
EOF
rm /etc/google-fluentd/google-fluentd.conf.bak

sudo service google-fluentd restart

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
    -e BATCH_LOGS_BUCKET_NAME=$BATCH_LOGS_BUCKET_NAME \
    -e WORKER_LOGS_BUCKET_NAME=$WORKER_LOGS_BUCKET_NAME \
    -e INSTANCE_ID=$INSTANCE_ID \
    -e PROJECT=$PROJECT \
    -e WORKER_CONFIG=$WORKER_CONFIG \
    -e MAX_IDLE_TIME_MSECS=$MAX_IDLE_TIME_MSECS \
    -e WORKER_DATA_DISK_MOUNT=/mnt/disks/$WORKER_DATA_DISK_NAME \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /usr/sbin/xfs_quota:/usr/sbin/xfs_quota \
    -v /batch:/batch \
    -v /logs:/logs \
    -v /gcsfuse:/gcsfuse:shared \
    -v /xfsquota:/xfsquota \
    --mount type=bind,source=/mnt/disks/$WORKER_DATA_DISK_NAME,target=/host \
    -p 5000:5000 \
    --device /dev/fuse \
    --device $XFS_DEVICE \
    --cap-add SYS_ADMIN \
    --security-opt apparmor:unconfined \
    $BATCH_WORKER_IMAGE \
    python3 -u -m batch.worker.worker >worker.log 2>&1

while true; do
  gcloud -q compute instances delete $NAME --zone=$ZONE
  sleep 1
done
'''
                }, {
                    'key': 'shutdown-script',
                    'value': '''
set -x

WORKER_LOGS_BUCKET_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker_logs_bucket_name")
INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_id")
NAME=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')

journalctl -u docker.service > dockerd.log

# this has to match LogStore.worker_log_path
gsutil -m cp dockerd.log gs://$WORKER_LOGS_BUCKET_NAME/batch/logs/$INSTANCE_ID/worker/$NAME/
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
                    'key': 'batch_logs_bucket_name',
                    'value': self.log_store.batch_logs_bucket_name
                }, {
                    'key': 'worker_logs_bucket_name',
                    'value': self.log_store.worker_logs_bucket_name
                }, {
                    'key': 'instance_id',
                    'value': self.log_store.instance_id
                }, {
                    'key': 'max_idle_time_msecs',
                    'value': max_idle_time_msecs
                }]
            },
            'tags': {
                'items': [
                    "batch2-agent"
                ]
            },
        }

        worker_config = WorkerConfig.from_instance_config(config)
        assert worker_config.is_valid_configuration(self.app['resources'])
        config['metadata']['items'].append({
            'key': 'worker_config',
            'value': base64.b64encode(json.dumps(worker_config.config).encode()).decode()
        })

        await self.compute_client.post(
            f'/zones/{zone}/instances', json=config)

        log.info(f'created machine {machine_name} for {instance} '
                 f' with logs at {self.log_store.worker_log_path(machine_name, "worker.log")}')

    async def call_delete_instance(self, instance, reason, timestamp=None, force=False):
        if instance.state == 'deleted' and not force:
            return
        if instance.state not in ('inactive', 'deleted'):
            await instance.deactivate(reason, timestamp)

        try:
            await self.compute_client.delete(
                f'/zones/{instance.zone}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
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
        payload = event.get('jsonPayload')
        if payload is None:
            log.warning('event has no payload')
            return

        timestamp = dateutil.parser.isoparse(event['timestamp']).timestamp() * 1000
        version = payload['version']
        if version != '1.2':
            log.warning('unknown event verison {version}')
            return

        resource_type = event['resource']['type']
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
        log.info('starting event loop')
        while True:
            try:
                row = await self.db.select_and_fetchone('SELECT * FROM `gevents_mark`;')
                if row['mark']:
                    mark = row['mark']
                else:
                    mark = datetime.datetime.utcnow().isoformat() + 'Z'

                filter = f'''
logName="projects/{PROJECT}/logs/compute.googleapis.com%2Factivity_log" AND
resource.type=gce_instance AND
jsonPayload.resource.name:"{self.machine_name_prefix}" AND
jsonPayload.event_subtype=("compute.instances.preempted" OR "compute.instances.delete")
AND timestamp >= "{mark}"
'''

                new_mark = None
                async for event in await self.logging_client.list_entries(
                        body={
                            'resourceNames': [f'projects/{PROJECT}'],
                            'orderBy': 'timestamp asc',
                            'pageSize': 100,
                            'filter': filter
                        }):
                    # take the last, largest timestamp
                    new_mark = event['timestamp']
                    await self.handle_event(event)

                if new_mark is not None:
                    await self.db.execute_update(
                        'UPDATE `gevents_mark` SET mark = %s;',
                        (new_mark,))
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception('in event loop')
            await asyncio.sleep(15)

    async def control_loop(self):
        log.info('starting control loop')
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
                        (ready_cores_mcpu - self.live_free_cores_mcpu + (self.worker_cores * 1000) - 1)
                        // (self.worker_cores * 1000))
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

                n_live_instances = self.n_instances_by_state['pending'] + self.n_instances_by_state['active']
                if ENABLE_STANDING_WORKER and n_live_instances == 0 and self.max_instances > 0:
                    await self.create_instance(cores=self.standing_worker_cores,
                                               max_idle_time_msecs=STANDING_WORKER_MAX_IDLE_TIME_MSECS)
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
            spec = await self.compute_client.get(
                f'/zones/{instance.zone}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                await self.remove_instance(instance, 'does_not_exist')
                return
            raise

        # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED
        gce_state = spec['status']
        log.info(f'{instance} gce_state {gce_state}')

        if gce_state in ('STOPPING', 'TERMINATED'):
            log.info(f'{instance} live but stopping or terminated, deactivating')
            await instance.deactivate('terminated')

        if (gce_state in ('STAGING', 'RUNNING')
                and instance.state == 'pending'
                and time_msecs() - instance.time_created > 5 * 60 * 1000):
            # FIXME shouldn't count time in PROVISIONING
            log.info(f'{instance} did not activate within 5m, deleting')
            await self.call_delete_instance(instance, 'activation_timeout')

        if instance.state == 'inactive':
            log.info(f'{instance} is inactive, deleting')
            await self.call_delete_instance(instance, 'inactive')

        await instance.update_timestamp()

    async def instance_monitoring_loop(self):
        log.info('starting instance monitoring loop')

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
