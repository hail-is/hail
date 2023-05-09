from hailtop.batch_client.parse import (
    CPU_REGEX,
    CPU_REGEXPAT,
    MEMORY_REGEX,
    MEMORY_REGEXPAT,
    STORAGE_REGEX,
    STORAGE_REGEXPAT,
)
from hailtop.utils.validate import (
    ValidationError,
    anyof,
    bool_type,
    dictof,
    int_type,
    keyed,
    listof,
    non_empty_str_type,
    nullable,
    numeric,
    oneof,
    regex,
    required,
    str_type,
    switch,
)

from ..globals import memory_types

k8s_str = regex(r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*', maxlen=253)

# FIXME validate image
# https://github.com/docker/distribution/blob/master/reference/regexp.go#L68
image_str = str_type


# DEPRECATED:
# command -> process/command
# image -> process/image
# mount_docker_socket -> process/mount_docker_socket
# pvc_size -> resources/storage
# gcsfuse -> cloudfuse


job_validator = keyed(
    {
        'always_copy_output': bool_type,
        'always_run': bool_type,
        'attributes': dictof(str_type),
        'env': listof(keyed({'name': str_type, 'value': str_type})),
        'cloudfuse': listof(
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
        'parent_ids': listof(int_type),
        'absolute_parent_ids': listof(int_type),
        'in_update_parent_ids': listof(int_type),
        'port': int_type,
        required('process'): switch(
            'type',
            {
                'docker': {
                    required('command'): listof(str_type),
                    required('image'): image_str,
                    'mount_docker_socket': bool_type,  # DEPRECATED
                },
                'jvm': {
                    required('jar_spec'): keyed(
                        {required('type'): oneof('git_revision', 'jar_url'), required('value'): str_type}
                    ),
                    required('command'): listof(str_type),
                },
            },
        ),
        'regions': listof(str_type),
        'requester_pays_project': str_type,
        'resources': keyed(
            {
                'memory': anyof(regex(MEMORY_REGEXPAT, MEMORY_REGEX), oneof(*memory_types)),
                'cpu': regex(CPU_REGEXPAT, CPU_REGEX),
                'storage': regex(STORAGE_REGEXPAT, STORAGE_REGEX),
                'machine_type': str_type,
                'preemptible': bool_type,
            }
        ),
        'secrets': listof(
            keyed({required('namespace'): k8s_str, required('name'): k8s_str, required('mount_path'): str_type})
        ),
        'service_account': keyed({required('namespace'): k8s_str, required('name'): k8s_str}),
        'timeout': numeric(**{"x > 0": lambda x: x > 0}),
        'user_code': str_type,
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

batch_update_validator = keyed(
    {
        required('token'): str_type,
        required('n_jobs'): numeric(**{"x > 0": lambda x: isinstance(x, int) and x > 0}),
    }
)


def validate_and_clean_jobs(jobs):
    if not isinstance(jobs, list):
        raise ValidationError('jobs is not list')
    for i, job in enumerate(jobs):
        handle_deprecated_job_keys(i, job)
        job_validator.validate(f"jobs[{i}]", job)
        handle_job_backwards_compatibility(job)


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
        process_keys = ['command', 'image']
        if 'command' not in job or 'image' not in job:
            raise ValidationError(
                f'jobs[{i}].process is not defined, but '
                f'deprecated keys {[k for k in process_keys if k not in job]} '
                f'are not in jobs[{i}]'
            )
        command = job['command']
        image = job['image']
        try:
            for k in process_keys:
                job_validator['process']['docker'][k].validate(f"jobs[{i}].{k}", job[k])
        except ValidationError as e:
            raise ValidationError(
                f"[command, image keys are "
                f"DEPRECATED. Use process.command, process.image, "
                f"with process.type = 'docker'.] "
                f"{e.reason}"
            ) from e

        job['process'] = {
            'command': command,
            'image': image,
            'type': 'docker',
        }
        del job['command']
        del job['image']
    elif 'command' in job or 'image' in job:
        raise ValidationError(
            f"jobs[{i}].process is already defined, but "
            f"deprecated keys 'command', 'image' "
            f"are also present. "
            f"Please remove deprecated keys."
        )

    mount_docker_socket = job['process'].pop('mount_docker_socket', False)
    if mount_docker_socket:
        raise ValidationError(
            "mount_docker_socket is no longer supported but was set to True in request. Please upgrade."
        )

    if 'gcsfuse' in job:
        job['cloudfuse'] = job.pop('gcsfuse')


def handle_job_backwards_compatibility(job):
    if 'cloudfuse' in job:
        job['gcsfuse'] = job.pop('cloudfuse')
    if 'parent_ids' in job:
        job['absolute_parent_ids'] = job.pop('parent_ids')
    if 'always_copy_output' not in job:
        job['always_copy_output'] = True


def validate_batch(batch):
    batch_validator.validate('batch', batch)


def validate_batch_update(update):
    batch_update_validator.validate('batch_update', update)
