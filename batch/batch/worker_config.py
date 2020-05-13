import re

from .globals import WORKER_CONFIG_VERSION

MACHINE_TYPE_REGEX = re.compile('projects/([^/]+)/zones/([^/]+)/machineTypes/([^-]+)-([^-]+)-(\d+)')
DISK_TYPE_REGEX = re.compile('projects/([^/]+)/zones/([^/]+)/diskTypes/(.+)')


def parse_machine_type_str(name):
    match = MACHINE_TYPE_REGEX.fullmatch(name)
    assert match and len(match.groups()) == 5
    return match.groups()


def parse_disk_type(name):
    match = DISK_TYPE_REGEX.fullmatch(name)
    assert match and len(match.groups()) == 3
    return match.groups()


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
# boot_disk: dict
#   project: str
#   zone: str
#   type_: str (pd-ssd, pd-standard, local-ssd)
#   size: int (in GB)
#   image: str


class WorkerConfig:
    @staticmethod
    def from_instance_config(instance_config):
        instance_project, instance_zone, instance_family, instance_type, instance_cores = \
            parse_machine_type_str(instance_config['machineType'])

        preemptible = instance_config['scheduling']['preemptible']

        boot_disk = [disk_config for disk_config in instance_config['disks']
                     if disk_config['boot']][0]

        boot_disk_project, boot_disk_zone, boot_disk_type = parse_disk_type(boot_disk['initializeParams']['diskType'])
        boot_disk_image = boot_disk['initializeParams']['sourceImage']
        boot_disk_size = int(boot_disk['initializeParams']['diskSizeGb'])

        config = {
            'version': WORKER_CONFIG_VERSION,
            'instance': {
                'project': instance_project,
                'zone': instance_zone,
                'family': instance_family,
                'type': instance_type,
                'cores': int(instance_cores),
                'preemptible': preemptible
            },
            'boot_disk': {
                'project': boot_disk_project,
                'zone': boot_disk_zone,
                'type': boot_disk_type,
                'size': boot_disk_size,
                'image': boot_disk_image
            }
        }

        return WorkerConfig(config)

    def __init__(self, config):
        self.config = config

        self.version = self.config['version']
        assert self.version == 1

        instance = self.config['instance']
        boot_disk = self.config['boot_disk']

        self.instance_project = instance['project']
        self.instance_zone = instance['zone']
        self.instance_family = instance['family']
        self.instance_type = instance['type']

        self.cores = instance['cores']
        self.preemptible = instance['preemptible']
        self.boot_disk_size_gb = boot_disk['size']
        self.boot_disk_type = boot_disk['type']

    def is_valid_configuration(self, valid_resources):
        is_valid = True
        dummy_resources = self.resources(0, 0)
        for resource in dummy_resources:
            is_valid &= resource['name'] in valid_resources
        return is_valid

    def resources(self, cpu_in_mcpu, memory_in_bytes):
        resources = []

        preemptible = 'preemptible' if self.preemptible else 'nonpreemptible'
        worker_fraction_in_1024ths = 1024 * cpu_in_mcpu // (self.cores * 1000)

        resources.append({'name': f'compute/{self.instance_family}-{self.instance_type}-{preemptible}/1',
                          'quantity': cpu_in_mcpu})

        resources.append({'name': f'memory/{self.instance_family}-{self.instance_type}-{preemptible}/1',
                          'quantity': memory_in_bytes / 1024 / 1024})

        # the factors of 1024 cancel between GiB -> MiB and fraction_1024 -> fraction
        resources.append({'name': f'boot-disk/{self.boot_disk_type}/1',
                          'quantity': self.boot_disk_size_gb * worker_fraction_in_1024ths})

        resources.append({'name': 'service-fee/1',
                          'quantity': cpu_in_mcpu})

        if is_power_two(self.cores) and self.cores <= 256:
            resources.append({'name': 'ip-fee/1024/1',
                              'quantity': worker_fraction_in_1024ths})
        else:
            raise NotImplementedError(self.cores)

        return resources
