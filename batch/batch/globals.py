states = {
    'Pending',
    'Ready',
    'Creating',
    'Running',
    'Cancelled',
    'Error',
    'Failed',
    'Success',
}

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

memory_types = ('lowmem', 'standard', 'highmem')

HTTP_CLIENT_MAX_SIZE = 8 * 1024 * 1024

BATCH_FORMAT_VERSION = 7
STATUS_FORMAT_VERSION = 5
INSTANCE_VERSION = 26

MAX_PERSISTENT_SSD_SIZE_GIB = 64 * 1024
RESERVED_STORAGE_GB_PER_CORE = 5
