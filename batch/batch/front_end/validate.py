from hailtop.batch_client.parse import (
    MEMORY_REGEX,
    MEMORY_REGEXPAT,
    CPU_REGEX,
    CPU_REGEXPAT,
    STORAGE_REGEX,
    STORAGE_REGEXPAT,
)

from hailtop.utils.validate import (
    anyof,
    bool_type,
    dictof,
    keyed,
    listof,
    int_type,
    nullable,
    numeric,
    oneof,
    regex,
    required,
    str_type,
    non_empty_str_type,
    switch,
    ValidationError,
)

from ..globals import valid_machine_types, memory_to_worker_type

k8s_str = regex(r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*', maxlen=253)

# FIXME validate image
# https://github.com/docker/distribution/blob/master/reference/regexp.go#L68
image_str = str_type


# DEPRECATED:
# command -> process/command
# image -> process/image
# mount_docker_socket -> process/mount_docker_socket
# pvc_size -> resources/storage


job_validator = keyed(
    {
        'always_run': bool_type,
        'attributes': dictof(str_type),
        'env': listof(keyed({'name': str_type, 'value': str_type})),
        'gcsfuse': listof(
            keyed(
                {
                    required('bucket'): non_empty_str_type,
                    required('mount_path'): non_empty_str_type,
                    required('read_only'): bool_type,
                }
            )
        ),
        'input_files': listof(keyed({required('from'): str_type, required('to'): str_type})),
        required('job_id'): int_type,
        'mount_tokens': bool_type,
        'network': oneof('public', 'private'),
        'unconfined': bool_type,
        'output_files': listof(keyed({required('from'): str_type, required('to'): str_type})),
        required('parent_ids'): listof(int_type),
        'port': int_type,
        required('process'): switch(
            'type',
            {
                'docker': {
                    required('command'): listof(str_type),
                    required('image'): image_str,
                    required('mount_docker_socket'): bool_type,
                },
                'jvm': {required('command'): listof(str_type)},
            },
        ),
        'requester_pays_project': str_type,
        'resources': keyed(
            {
                'memory': anyof(regex(MEMORY_REGEXPAT, MEMORY_REGEX), oneof(*memory_to_worker_type.keys())),
                'cpu': regex(CPU_REGEXPAT, CPU_REGEX),
                'storage': regex(STORAGE_REGEXPAT, STORAGE_REGEX),
                'machine_type': oneof(*valid_machine_types),
                'preemptible': bool_type,
            }
        ),
        'secrets': listof(
            keyed({required('namespace'): k8s_str, required('name'): k8s_str, required('mount_path'): str_type})
        ),
        'service_account': keyed({required('namespace'): k8s_str, required('name'): k8s_str}),
        'timeout': numeric(**{"x > 0": lambda x: x > 0}),
    }
)

batch_validator = keyed(
    {
        'attributes': nullable(dictof(str_type)),
        required('billing_project'): str_type,
        'callback': nullable(str_type),
        required('n_jobs'): int_type,
        required('token'): str_type,
        'cancel_after_n_failures': nullable(numeric(**{"x > 0": lambda x: isinstance(x, int) and x > 0})),
    }
)


def validate_and_clean_jobs(jobs):
    if not isinstance(jobs, list):
        raise ValidationError('jobs is not list')
    for i, job in enumerate(jobs):
        handle_deprecated_job_keys(i, job)
        job_validator.validate(f"jobs[{i}]", job)


def handle_deprecated_job_keys(i, job):
    if 'pvc_size' in job:
        if 'resources' in job and 'storage' in job['resources']:
            raise ValidationError(
                f"jobs[{i}].resources.storage is already defined, but " f"deprecated key 'pvc_size' is also present."
            )
        pvc_size = job['pvc_size']
        try:
            job_validator['resources']['storage'].validate(f"jobs[{i}].pvc_size", job['pvc_size'])
        except ValidationError as e:
            raise ValidationError(f"[pvc_size key is DEPRECATED. Use " f"resources.storage] {e.reason}") from e
        resources = job.get('resources')
        if resources is None:
            resources = {}
            job['resources'] = resources
        resources['storage'] = pvc_size
        del job['pvc_size']

    if 'process' not in job:
        process_keys = ['command', 'image', 'mount_docker_socket']
        if 'command' not in job or 'image' not in job or 'mount_docker_socket' not in job:
            raise ValidationError(
                f'jobs[{i}].process is not defined, but '
                f'deprecated keys {[k for k in process_keys if k not in job]} '
                f'are not in jobs[{i}]'
            )
        command = job['command']
        image = job['image']
        mount_docker_socket = job['mount_docker_socket']
        try:
            for k in process_keys:
                job_validator['process']['docker'][k].validate(f"jobs[{i}].{k}", job[k])
        except ValidationError as e:
            raise ValidationError(
                f"[command, image, mount_docker_socket keys are "
                f"DEPRECATED. Use process.command, process.image, "
                f"process.mount_docker_socket with process.type = 'docker'.] "
                f"{e.reason}"
            ) from e

        job['process'] = {
            'command': command,
            'image': image,
            'mount_docker_socket': mount_docker_socket,
            'type': 'docker',
        }
        del job['command']
        del job['image']
        del job['mount_docker_socket']
    else:
        if 'command' in job or 'image' in job or 'mount_docker_socket' in job:
            raise ValidationError(
                f"jobs[{i}].process is already defined, but "
                f"deprecated keys 'command', 'image', "
                f"'mount_docker_socket' are also present. "
                f"Please remove deprecated keys."
            )


def validate_batch(batch):
    batch_validator.validate('batch', batch)
