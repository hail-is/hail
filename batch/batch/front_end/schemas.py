batch_schema = {
    'attributes': {
        'type': 'dict',
        'keyschema': {'type': 'string', 'nullable': False},
        'valueschema': {'type': 'string', 'nullable': False}
    },
    'billing_project': {'type': 'string', 'required': True},
    'callback': {'type': 'string', 'nullable': True},
    'n_jobs': {'type': 'integer', 'required': True},
    'token': {'type': 'string', 'required': True}
}
