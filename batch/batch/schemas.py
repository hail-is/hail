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
        'spec': pod_spec,
        'job_id': {'required': True, 'type': 'integer'},
        'parent_ids': {'type': 'list', 'schema': {'type': 'integer'}},
        'input_files': {
            'type': 'list',
            'schema': {'type': 'list', 'items': 2 * ({'type': 'string'},)}},
        'output_files': {
            'type': 'list',
            'schema': {'type': 'list', 'items': 2 * ({'type': 'string'},)}},
        'always_run': {'type': 'boolean'},
        'pvc_size': {'type': 'string'},
        'attributes': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        },
        'callback': {'type': 'string'}
    }
}

batch_schema = {
    'attributes': {
        'type': 'dict',
        'keyschema': {'type': 'string'},
        'valueschema': {'type': 'string'}
    },
    'callback': {'type': 'string'},
    'jobs': {'type': 'list', 'schema': job_schema, 'nullable': False}
}