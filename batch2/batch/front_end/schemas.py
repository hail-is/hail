batch_schema = {
    'attributes': {
        'type': 'dict',
        'keyschema': {'type': 'string', 'nullable': False},
        'valueschema': {'type': 'string', 'nullable': False}
    },
    'billing_project': {'type': 'string', 'nullable': False},
    'callback': {'type': 'string'},
    'n_jobs': {'type': 'integer', 'nullable': False}
}
