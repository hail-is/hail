batch_schema = {
    'attributes': {
        'type': 'dict',
        'keyschema': {'type': 'string'},
        'valueschema': {'type': 'string'}
    },
    'callback': {'type': 'string'},
    'n_jobs': {'type': 'integer', 'nullable': False}
}
