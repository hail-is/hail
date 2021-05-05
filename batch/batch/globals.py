states = {'Pending', 'Ready', 'Creating', 'Running', 'Cancelled', 'Error', 'Failed', 'Success'}

complete_states = ('Cancelled', 'Error', 'Failed', 'Success')

valid_state_transitions = {
    'Pending': {'Ready'},
    'Ready': {'Creating', 'Running', 'Cancelled', 'Error'},
    'Creating': {'Ready', 'Running'},
    'Running': {'Ready', 'Cancelled', 'Error', 'Failed', 'Success'},
    'Cancelled': set(),
    'Error': set(),
    'Failed': set(),
    'Success': set(),
}

tasks = ('input', 'main', 'output')

valid_machine_types = []
for typ in ('highcpu', 'standard', 'highmem'):
    if typ == 'standard':
        possible_cores = [1, 2, 4, 8, 16, 32, 64, 96]
    else:
        possible_cores = [2, 4, 8, 16, 32, 64, 96]
    for cores in possible_cores:
        valid_machine_types.append(f'n1-{typ}-{cores}')

memory_to_worker_type = {'lowmem': 'highcpu', 'standard': 'standard', 'highmem': 'highmem'}

HTTP_CLIENT_MAX_SIZE = 8 * 1024 * 1024

BATCH_FORMAT_VERSION = 6
STATUS_FORMAT_VERSION = 5
INSTANCE_VERSION = 18
WORKER_CONFIG_VERSION = 3

MAX_PERSISTENT_SSD_SIZE_GIB = 64 * 1024
RESERVED_STORAGE_GB_PER_CORE = 5
