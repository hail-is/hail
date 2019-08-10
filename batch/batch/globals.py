
states = {'Pending', 'Running', 'Cancelled', 'Error', 'Failed', 'Success'}

complete_states = ('Cancelled', 'Error', 'Failed', 'Success')

valid_state_transitions = {
    'Pending': {'Running'},
    'Running': {'Running', 'Cancelled', 'Error', 'Failed', 'Success'},
    'Cancelled': set(),
    'Error': set(),
    'Failed': set(),
    'Success': set(),
}
