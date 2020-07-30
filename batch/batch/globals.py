states = {'Pending', 'Ready', 'Running', 'Cancelled', 'Error', 'Failed', 'Success'}

complete_states = ('Cancelled', 'Error', 'Failed', 'Success')

valid_state_transitions = {
    'Pending': {'Ready'},
    'Ready': {'Running', 'Cancelled', 'Error'},
    'Running': {'Ready', 'Cancelled', 'Error', 'Failed', 'Success'},
    'Cancelled': set(),
    'Error': set(),
    'Failed': set(),
    'Success': set(),
}

tasks = ('input', 'main', 'output')

HTTP_CLIENT_MAX_SIZE = 8 * 1024 * 1024

BATCH_FORMAT_VERSION = 3
STATUS_FORMAT_VERSION = 3
INSTANCE_VERSION = 9
WORKER_CONFIG_VERSION = 2
