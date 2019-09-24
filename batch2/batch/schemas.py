pod_spec = {
    'type': 'dict',
    'required': True,
    'allow_unknown': True,
    'schema': {}
}

job_schema = {
    'type': 'dict',
    'required': True,
    'allow_unknown': True,
    'schema': {
        'job_id': {'type': 'integer', 'nullable': False},
        'spec': pod_spec,
        'parent_ids': {'type': 'list', 'schema': {'type': 'integer'}},
        'input_files': {
            'type': 'list',
            'schema': {'type': 'list', 'items': 2 * ({'type': 'string'},)}},
        'output_files': {
            'type': 'list',
            'schema': {'type': 'list', 'items': 2 * ({'type': 'string'},)}},
        'always_run': {'type': 'boolean'},
        'attributes': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        },
        'callback': {'type': 'string'}
    }
}

job_array_schema = {
    'jobs': {'type': 'list', 'schema': job_schema}
}

batch_schema = {
    'attributes': {
        'type': 'dict',
        'keyschema': {'type': 'string'},
        'valueschema': {'type': 'string'}
    },
    'callback': {'type': 'string'}
}
