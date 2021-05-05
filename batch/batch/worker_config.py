import re
from collections import defaultdict

from .globals import WORKER_CONFIG_VERSION

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inst_coll_config import PoolConfig  # pylint: disable=cyclic-import

MACHINE_TYPE_REGEX = re.compile(
    'projects/(?P<project>[^/]+)/zones/(?P<zone>[^/]+)/machineTypes/(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)'
)
DISK_TYPE_REGEX = re.compile('(projects/(?P<project>[^/]+)/)?zones/(?P<zone>[^/]+)/diskTypes/(?P<disk_type>.+)')


def parse_machine_type_str(name):
    match = MACHINE_TYPE_REGEX.fullmatch(name)
    return match.groupdict()


def parse_disk_type(name):
    match = DISK_TYPE_REGEX.fullmatch(name)
    return match.groupdict()


def is_power_two(n):
    return (n & (n - 1) == 0) and n != 0


# worker_config spec
#
# version: int
# instance: dict
#   project: str
#   zone: str
#   family: str (n1, n2, c2, e2, n2d, m1, m2)
#   type_: str (standard, highmem, highcpu)
#   cores: int
#   preemptible: bool
# disks: list of dict
#   boot: bool
#   project: str
#   zone: str
#   type_: str (pd-ssd, pd-standard, local-ssd)
#   size: int (in GB)


class WorkerConfig:
    @staticmethod
    def from_instance_config(instance_config, job_private=False):
        instance_info = parse_machine_type_str(instance_config['machineType'])

        preemptible = instance_config['scheduling']['preemptible']

        disks = []
        for disk_config in instance_config['disks']:
            params = disk_config['initializeParams']
            disk_info = parse_disk_type(params['diskType'])
            disk_type = disk_info['disk_type']

            if disk_type == 'local-ssd':
                disk_size = 375
            else:
                disk_size = int(params['diskSizeGb'])

            disks.append(
                {
                    'boot': disk_config.get('boot', False),
                    'project': disk_info.get('project'),
                    'zone': disk_info['zone'],
                    'type': disk_type,
                    'size': disk_size,
                    'image': params.get('sourceImage', None),
                }
            )

        config = {
            'version': WORKER_CONFIG_VERSION,
            'instance': {
                'project': instance_info['project'],
                'zone': instance_info['zone'],
                'family': instance_info['machine_family'],
                'type': instance_info['machine_type'],
                'cores': int(instance_info['cores']),
                'preemptible': preemptible,
            },
            'disks': disks,
            'job-private': job_private,
        }

        return WorkerConfig(config)

    @staticmethod
    def from_pool_config(pool_config: 'PoolConfig'):
        disks = [
            {
                'boot': True,
                'project': None,
                'zone': None,
                'type': 'pd-ssd',
                'size': pool_config.boot_disk_size_gb,
                'image': None,
            }
        ]

        if pool_config.worker_local_ssd_data_disk:
            typ = 'local-ssd'
            size = 375
        else:
            typ = 'pd-ssd'
            size = pool_config.worker_pd_ssd_data_disk_size_gb

        disks.append({'boot': False, 'project': None, 'zone': None, 'type': typ, 'size': size, 'image': None})

        config = {
            'version': WORKER_CONFIG_VERSION,
            'instance': {
                'project': None,
                'zone': None,
                'family': 'n1',  # FIXME: need to figure out how to handle variable family types
                'type': pool_config.worker_type,
                'cores': pool_config.worker_cores,
                'preemptible': True,
            },
            'disks': disks,
            'job-private': False,
        }

        return WorkerConfig(config)

    def __init__(self, config):
        self.config = config

        self.version = self.config['version']
        assert self.version == 3

        instance = self.config['instance']
        self.disks = self.config['disks']

        self.instance_project = instance['project']
        self.instance_zone = instance['zone']
        self.instance_family = instance['family']
        self.instance_type = instance['type']

        self.cores = instance['cores']
        self.preemptible = instance['preemptible']

        assert len(self.disks) == 2
        boot_disk = self.disks[0]
        assert boot_disk['boot']
        data_disk = self.disks[1]
        assert not data_disk['boot']

        self.local_ssd_data_disk = data_disk['type'] == 'local-ssd'
        self.data_disk_size_gb = data_disk['size']

        self.job_private = self.config['job-private']

    def is_valid_configuration(self, valid_resources):
        is_valid = True
        dummy_resources = self.resources(0, 0, 0)
        for resource in dummy_resources:
            is_valid &= resource['name'] in valid_resources
        return is_valid

    def resources(self, cpu_in_mcpu, memory_in_bytes, storage_in_gib):
        assert memory_in_bytes % (1024 * 1024) == 0, memory_in_bytes
        assert isinstance(storage_in_gib, int), storage_in_gib

        resources = []

        preemptible = 'preemptible' if self.preemptible else 'nonpreemptible'
        worker_fraction_in_1024ths = 1024 * cpu_in_mcpu // (self.cores * 1000)

        resources.append({'name': f'compute/{self.instance_family}-{preemptible}/1', 'quantity': cpu_in_mcpu})

        resources.append(
            {'name': f'memory/{self.instance_family}-{preemptible}/1', 'quantity': memory_in_bytes // 1024 // 1024}
        )

        # storage is in units of MiB
        resources.append({'name': 'disk/pd-ssd/1', 'quantity': storage_in_gib * 1024})

        quantities = defaultdict(lambda: 0)
        for disk in self.disks:
            name = f'disk/{disk["type"]}/1'
            # the factors of 1024 cancel between GiB -> MiB and fraction_1024 -> fraction
            disk_size_in_mib = disk['size'] * worker_fraction_in_1024ths
            quantities[name] += disk_size_in_mib

        for name, quantity in quantities.items():
            resources.append({'name': name, 'quantity': quantity})

        resources.append({'name': 'service-fee/1', 'quantity': cpu_in_mcpu})

        if is_power_two(self.cores) and self.cores <= 256:
            resources.append({'name': 'ip-fee/1024/1', 'quantity': worker_fraction_in_1024ths})
        else:
            raise NotImplementedError(self.cores)

        return resources

    def cost_per_hour(self, resource_rates, cpu_in_mcpu, memory_in_bytes, storage_in_gb):
        resources = self.resources(cpu_in_mcpu, memory_in_bytes, storage_in_gb)
        cost_per_msec = 0
        for r in resources:
            name = r['name']
            quantity = r['quantity']
            rate_unit_msec = resource_rates[name]
            cost_per_msec += quantity * rate_unit_msec
        return cost_per_msec * 1000 * 60 * 60
