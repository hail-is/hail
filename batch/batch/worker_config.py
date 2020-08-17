import re
from collections import defaultdict

from .globals import WORKER_CONFIG_VERSION

MACHINE_TYPE_REGEX = re.compile('projects/(?P<project>[^/]+)/zones/(?P<zone>[^/]+)/machineTypes/(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)')
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
    def from_instance_config(instance_config):
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

            disks.append({
                'boot': disk_config.get('boot', False),
                'project': disk_info.get('project'),
                'zone': disk_info['zone'],
                'type': disk_type,
                'size': disk_size,
                'image': params.get('sourceImage', None)
            })

        config = {
            'version': WORKER_CONFIG_VERSION,
            'instance': {
                'project': instance_info['project'],
                'zone': instance_info['zone'],
                'family': instance_info['machine_family'],
                'type': instance_info['machine_type'],
                'cores': int(instance_info['cores']),
                'preemptible': preemptible
            },
            'disks': disks
        }

        return WorkerConfig(config)

    def __init__(self, config):
        self.config = config

        self.version = self.config['version']
        assert self.version == 2

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

        self.local_ssd_data_disk = (data_disk['type'] == 'local-ssd')
        self.data_disk_size_gb = data_disk['size']

    def is_valid_configuration(self, valid_resources):
        is_valid = True
        dummy_resources = self.resources(0, 0)
        for resource in dummy_resources:
            is_valid &= resource['name'] in valid_resources
        return is_valid

    def resources(self, cpu_in_mcpu, memory_in_bytes):
        assert memory_in_bytes % (1024 * 1024) == 0
        resources = []

        preemptible = 'preemptible' if self.preemptible else 'nonpreemptible'
        worker_fraction_in_1024ths = 1024 * cpu_in_mcpu // (self.cores * 1000)

        resources.append({'name': f'compute/{self.instance_family}-{preemptible}/1',
                          'quantity': cpu_in_mcpu})

        resources.append({'name': f'memory/{self.instance_family}-{preemptible}/1',
                          'quantity': memory_in_bytes // 1024 // 1024})

        quantities = defaultdict(lambda: 0)
        for disk in self.disks:
            name = f'disk/{disk["type"]}/1'
            # the factors of 1024 cancel between GiB -> MiB and fraction_1024 -> fraction
            disk_size_in_mib = disk['size'] * worker_fraction_in_1024ths
            quantities[name] += disk_size_in_mib

        for name, quantity in quantities.items():
            resources.append({'name': name,
                              'quantity': quantity})

        resources.append({'name': 'service-fee/1',
                          'quantity': cpu_in_mcpu})

        if is_power_two(self.cores) and self.cores <= 256:
            resources.append({'name': 'ip-fee/1024/1',
                              'quantity': worker_fraction_in_1024ths})
        else:
            raise NotImplementedError(self.cores)

        return resources
