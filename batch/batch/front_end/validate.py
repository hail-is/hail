import re

from hailtop.batch_client.parse import (MEMORY_REGEX, MEMORY_REGEXPAT,
                                        CPU_REGEX, CPU_REGEXPAT)

class required:
    def __init__(self, wrapped):
        self.wrapped = wrapped


class TypedValidator:
    def __init__(self, t):
        self.t = t

    def validate(self, name, obj):
        if not isinstance(obj, self.t):
            raise ValidationError(f'{name} is not {self.t}')


class dictof(TypedValidator):
    def __init__(self, vchecker):
        super().__init__(dict)
        self.vchecker = vchecker

    def validate(self, name, obj):
        super().validate(name, obj)
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValidationError(f'{name} has non-str key')
            self.vchecker.validate(f"{name}[{k}]", obj[k])


class keyed(TypedValidator):
    def __init__(self, keyed_checkers):
        super().__init__(dict)
        self.checkers = keyed_checkers

    def __getitem__(self, key):
        return self.checkers[key]

    def validate(self, name, obj):
        super().validate(name, obj)
        for k in obj:
            if k not in self.checkers:
                raise ValidationError(f'unknown key in {name}: {k}')
        for k, checker in self.checkers.items():
            if isinstance(checker, required):
                if k not in obj:
                    raise ValidationError(f'{name} missing required key {k}.')
                else:
                    checker.wrapped.validate(f"{name}.{key}", obj[k])
            elif k in obj:
                checker.validate(f"{name}.{key}", obj[k])


class listof(TypedValidator):
    def __init__(self, checker):
        super().__init__(list)
        self.checker = checker

    def validate(self, name, obj):
        super().validate(name, obj)
        for i, elt in enumerate(obj):
            self.checker.validate(f"{name}[{i}]", elt)


class in_set:
    def __init__(self, *valid):
        self.valid = set(valid)

    def validate(self, name, obj):
        if obj not in self.valid:
            raise ValidationError(f'{name} must be one of: {self.valid}')


class regex(TypedValidator):
    def __init__(self, pattern, re_obj=None, maxlen=None):
        super().__init__(str)
        self.pattern = pattern
        self.re_obj = re_obj if re_obj is not None else re.compile(pattern)
        self.maxlen = maxlen

    def validate(self, name, obj):
        super().validate(name, obj)
        if self.maxlen is not None and len(obj) > self.maxlen:
            raise ValidationError(f'length of {name} must be <= {self.maxlen}')
        if not self.re_obj.fullmatch(obj):
            raise ValidationError(f'{name} must match regex: {self.pattern}')


class numeric:
    def __init__(self, **conditions):
        self.conditions = conditions

    def validate(self, name, obj):
        if not isinstance(obj, int) and not isinstance(obj, float):
            raise ValidationError(f'{name} is not numeric')
        for text, condition in self.conditions.items():
            if not condition(obj):
                raise ValidationError(f'{name} does not satisfy the condition {text}')


class switch(TypedValidator):
    def __init__(self, key, checkers):
        super().__init__(dict)
        self.key = key
        self.valid_key = required(in_set(*checkers.keys()))
        self.checkers = {k: keyed({key: self.valid_key, **fields}) for k, fields in checkers.items()}

    def __getitem__(self, key):
        return self.checkers[key]

    def validate(self, name, obj):
        super().validate(name, obj)
        key = obj[self.key]
        self.valid_key.validate(f"{name}.{key}", key)
        self.checkers[key].validate(obj)


class nullable:
    def __init__(self, wrapped):
        self.checker = wrapped

    def validate(self, name, obj):
        if obj is not None:
            self.checker.validate(obj)


str_type = TypedValidator(str)
bool_type = TypedValidator(bool)
int_type = TypedValidator(int)

k8s_str = regex(r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*', maxlen=253)

# FIXME validate image
# https://github.com/docker/distribution/blob/master/reference/regexp.go#L68
image_str = str_type


class ValidationError(Exception):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason


def validate_and_clean_jobs(jobs):
    if not isinstance(jobs, list):
        raise ValidationError('jobs is not list')
    for i, job in enumerate(jobs):
        handle_deprecated_job_keys(i, job)
        job_validator.validate(f"jobs[{i}]", job)


def handle_deprecated_job_keys(i, job):
    if 'pvc_size' in job:
        if 'resources' in job and 'storage' in job['resources']:
            raise ValidationError(f"jobs[{i}].resources.storage is already defined, but "
                                  f"deprecated key 'pvc_size' is also present.")
        deprecated_msg = "[pvc_size key is DEPRECATED. Use resources.storage]"
        pvc_size = job['pvc_size']
        try:
            job_validator['resources']['storage'].validate(f"jobs[{i}].pvc_size", job['pvc_size'])
        except ValidationError as e:
            raise ValidationError(f"[pvc_size key is DEPRECATED. Use resources.storage] {e.reason}")
        resources = job.get('resources')
        if resources is None:
            resources = {}
            job['resources'] = resources
        resources['storage'] = pvc_size
        del job['pvc_size']

    if 'process' not in job:
        process_keys = ['command', 'image', 'mount_docker_socket']
        deprecated_msg = "[command, image, mount_docker_socket keys are DEPRECATED. " \
                         "Use process.command, process.image, process.mount_docker_socket " \
                         "with process.type = 'docker'.]"
        if 'command' not in job or 'image' not in job or 'mount_docker_socket' not in job:

            raise ValidationError(f'jobs[{i}].process is not defined, but deprecated keys {[k for k in process_keys if k not in job]} are not in jobs[{i}]')
        command = job['command']
        image = job['image']
        mount_docker_socket = job['mount_docker_socket']
        try:
            for k in process_keys:
                job_validator['process']['docker'][k].validate(f"jobs[{i}].{k}", job[k])
        except ValidationError as e:
            raise ValidationError(f"[command, image, mount_docker_socket keys are "
                                  f"DEPRECATED. Use process.command, process.image, "
                                  f"process.mount_docker_socket with process.type = 'docker'.] "
                                  f"{e.reason}")

        job['process'] = {'command': command,
                          'image': image,
                          'mount_docker_socket': mount_docker_socket,
                          'type': 'docker'}
        del job['command']
        del job['image']
        del job['mount_docker_socket']
    else:
        if 'command' in job or 'image' in job or 'mount_docker_socket' in job:
            raise ValidationError(f"jobs[{i}].process is already defined, but "
                                  f"deprecated keys 'command', 'image', "
                                  f"'mount_docker_socket' are also present. "
                                  f"Please remove deprecated keys.")


# DEPRECATED:
# command -> process/command
# image -> process/image
# mount_docker_socket -> process/mount_docker_socket
# pvc_size -> resources/storage

job_validator = keyed({
    'always_run': bool_type,
    'attributes': dictof(str_type),
    'env': listof(keyed({
        'name': str_type,
        'value': str_type
    })),
    'gcsfuse': listof(keyed({
        'bucket': required(str_type),
        'mount_path': required(str_type),
        'read_only': required(bool_type),
    })),
    'input_files': listof(keyed({
        'from': required(str_type),
        'to': required(str_type)
    })),
    'job_id': required(int_type),
    'mount_tokens': bool_type,
    'network': in_set('public', 'private'),
    'output_files': listof(keyed({
        'from': required(str_type),
        'to': required(str_type)
    })),
    'parent_ids': required(listof(int_type)),
    'port': int_type,
    'process': required(switch('type', {
        'docker': {
            'command': required(listof(str_type)),
            'image': required(image_str),
            'mount_docker_socket': required(bool_type)
        }
    })),
    'requester_pays_project': str_type,
    'resources': keyed({
        'memory': regex(MEMORY_REGEXPAT, MEMORY_REGEX),
        'cpu': regex(CPU_REGEXPAT, CPU_REGEX),
        'storage': regex(MEMORY_REGEXPAT, MEMORY_REGEX)
    }),
    'secrets': listof(keyed({
        'namespace': required(k8s_str),
        'name': required(k8s_str),
        'mount_path': required(str_type)
    })),
    'service_account': keyed({
        'namespace': required(k8s_str),
        'name': required(k8s_str)
    }),
    'timeout': numeric(**{"x > 0", lambda x: x > 0})
})

batch_validator = keyed({
    'attributes': nullable(dictof(str_type)),
    'billing_project': required(str_type),
    'callback': nullable(str),
    'n_jobs': required(int_type),
    'token': str_type
})

def validate_batch(batch):
    batch_validator.validate('batch', batch)