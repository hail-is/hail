
states = {'Pending', 'Ready', 'Running', 'Cancelled', 'Error', 'Failed', 'Success'}

complete_states = ('Cancelled', 'Error', 'Failed', 'Success')

valid_state_transitions = {
    'Pending': {'Ready', 'Cancelled'},
    'Ready': {'Pending', 'Running', 'Error', 'Cancelled'},
    'Running': set(complete_states),
    'Cancelled': set(),
    'Error': set(),
    'Failed': set(),
    'Success': set(),
}
