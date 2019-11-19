import re

# rough schema (without requiredness, value validation):
# jobs_schema = [{
#   'always_run': bool,
#   'attributes': {str: str},
#   'callback': str,
#   'command': [str],
#   'env': [{
#     'name': str,
#     'value': str
#   }],
#   'image': str,
#   'input_files': [{"from": str, "to": str}],
#   'job_id': int,
#   'mount_docker_socket': bool,
#   'output_files': [{"from": str, "to": str}],
#   'parent_ids': [int],
#   'pvc_size': str,
#   'resoures': {
#     'memory': str,
#     'cpu': str
#   },
#   secrets: [{
#     'namespace': str,
#     'name': str,
#     'mount_path': str
#   }],
#   service_account: {
#     'namespace': str,
#     'name': str
#   }
# }]

JOB_KEYS = {
    'always_run', 'attributes', 'callback', 'command', 'env', 'image', 'input_files', 'job_id', 'mount_docker_socket', 'output_files', 'parent_ids', 'pvc_size', 'resources', 'secrets', 'service_account'
}

ENV_VAR_KEYS = {'name', 'value'}

SECRET_KEYS = {'namespace', 'name', 'mount_path'}

RESOURCES_KEYS = {'memory', 'cpu'}

FILE_KEYS = {'from', 'to'}

K8S_NAME_REGEXPAT = r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*'
K8S_NAME_REGEX = re.compile(K8S_NAME_REGEXPAT)

MEMORY_REGEXPAT = r'[+]?(?:[0-9]*[.])?[0-9]+(?:[KMGTP][i]?)?'
MEMORY_REGEX = re.compile(MEMORY_REGEXPAT)

CPU_REGEXPAT = r'[+]?(?:[0-9]*[.])?[0-9]+[m]?'
CPU_REGEX = re.compile(CPU_REGEXPAT)


class ValidationError(Exception):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason


def validate_jobs(jobs):
    if not isinstance(jobs, list):
        raise ValidationError('jobs is not list')
    for i, job in enumerate(jobs):
        validate_job(i, job)


def validate_job(i, job):
    if not isinstance(job, dict):
        raise ValidationError(f'jobs[{i}] not dict')

    for k in job:
        if k not in JOB_KEYS:
            raise ValidationError(f'unknown key in jobs[{i}]: {k}')

    if 'always_run' in job:
        always_run = job['always_run']
        if not isinstance(always_run, bool):
            raise ValidationError(f'jobs[{i}].always_run is not bool')

    if 'attributes' in job:
        attributes = job['attributes']
        if not isinstance(attributes, dict):
            raise ValidationError(f'jobs[{i}].attributes is not dict')
        for k, v in attributes.items():
            if not isinstance(k, str):
                raise ValidationError(f'jobs[{i}].attributes has non-str key')
            if not isinstance(v, str):
                raise ValidationError(f'jobs[{i}].attributes has non-str value')

    if 'callback' in job:
        callback = job['callback']
        if not isinstance(callback, str):
            raise ValidationError(f'jobs[{i}].callback not str')

    if 'command' not in job:
        raise ValidationError(f'no required key command in jobs[{i}]')
    command = job['command']
    if not isinstance(command, list):
        raise ValidationError(f'jobs[{i}].command not list')
    for j, a in enumerate(command):
        if not isinstance(a, str):
            raise ValidationError(f'jobs[{i}].command[{j}] is not str')

    if 'env' in job:
        env = job['env']
        if not isinstance(env, list):
            raise ValidationError(f'jobs[{i}].env is not list')
        for j, e in enumerate(env):
            if not isinstance(e, dict):
                raise ValidationError(f'jobs[{i}].env[{j}] is not dict')
            for k in e:
                if k not in ENV_VAR_KEYS:
                    raise ValidationError(f'unknown key in jobs[{i}].env[{j}]: {k}')
            if 'name' not in e:
                raise ValidationError(f'no required key name in jobs[{i}].env[{j}]')
            name = e['name']
            if not isinstance(name, str):
                raise ValidationError(f'jobs[{i}].env[{j}].name is not str')
            if 'value' not in e:
                raise ValidationError(f'no required key value in jobs[{i}].env[{j}]')
            value = e['value']
            if not isinstance(value, str):
                raise ValidationError(f'jobs[{i}].env[{j}].value is not str')

    if 'image' not in job:
        raise ValidationError(f'no required key image in jobs[{i}]')
    image = job['image']
    if not isinstance(image, str):
        raise ValidationError(f'jobs[{i}].image is not str')
    # FIXME validate image
    # https://github.com/docker/distribution/blob/master/reference/regexp.go#L68

    if 'input_files' in job:
        input_files = job['input_files']
        if not isinstance(input_files, list):
            raise ValidationError(f'jobs[{i}].input_files not list')

        for j, f in enumerate(input_files):
            if not isinstance(f, dict):
                raise ValidationError(f'jobs[{i}].input_files[{j}] not dict')
            for k in f:
                if k not in FILE_KEYS:
                    raise ValidationError(f'unknown key in jobs[{i}].input_files[{j}]: {k}')

                if 'from' not in f:
                    raise ValidationError(f'no required key from in jobs[{i}].input_files[{j}]')
                src = f['from']
                if not isinstance(src, str):
                    raise ValidationError(f'jobs[{i}].input_files[{j}].from is not str')

                if 'to' not in f:
                    raise ValidationError(f'no required key to in jobs[{i}].input_files[{j}]')
                dst = f['to']
                if not isinstance(dst, str):
                    raise ValidationError(f'jobs[{i}].input_files[{j}].to is not str')

    if 'job_id' not in job:
        raise ValidationError(f'no required key job_id in jobs[{i}]')
    job_id = job['job_id']
    if not isinstance(job_id, int):
        raise ValidationError(f'jobs[{i}].job_id is not int')

    if 'mount_docker_socket' not in job:
        raise ValidationError(f'no required key mount_docker_socket in jobs[{i}]')
    mount_docker_socket = job['mount_docker_socket']
    if not isinstance(mount_docker_socket, bool):
        raise ValidationError(f'jobs[{i}].mount_docker_socket not bool')

    if 'output_files' in job:
        output_files = job['output_files']
        if not isinstance(output_files, list):
            raise ValidationError(f'jobs[{i}].output_files not list')

        for j, f in enumerate(output_files):
            if not isinstance(f, dict):
                raise ValidationError(f'jobs[{i}].output_files[{j}] not dict')

            for k in f:
                if k not in FILE_KEYS:
                    raise ValidationError(f'unknown key in jobs[{i}].output_files[{j}]: {k}')

            if 'from' not in f:
                raise ValidationError(f'no required key from in jobs[{i}].output_files[{j}]')
            src = f['from']
            if not isinstance(src, str):
                raise ValidationError(f'jobs[{i}].output_files[{j}].from is not str')

            if 'to' not in f:
                raise ValidationError(f'no required key to in jobs[{i}].output_files[{j}]')
            dst = f['to']
            if not isinstance(dst, str):
                raise ValidationError(f'jobs[{i}].output_files[{j}].to is not str')

    if 'parent_ids' not in job:
        raise ValidationError(f'no required key parent_ids in jobs[{i}]')
    parent_ids = job['parent_ids']
    if not isinstance(parent_ids, list):
        raise ValidationError(f'jobs[{i}].job_id is not list')
    for j, id in enumerate(parent_ids):
        if not isinstance(id, int):
            raise ValidationError(f'jobs[{i}].parent_ids[{j} is not int')

    if 'pvc_size' in job:
        pvc_size = job['pvc_size']
        if not isinstance(pvc_size, str):
            raise ValidationError(f'jobs[{i}].pvc_size not str')
        if not MEMORY_REGEX.fullmatch(pvc_size):
            raise ValidationError(f'jobs[{i}].pvc_size must match regex: {MEMORY_REGEXPAT}')

    if 'resources' in job:
        resources = job['resources']
        if not isinstance(resources, dict):
            raise ValidationError(f'jobs[{i}].resources is not dict')
        for k in resources:
            if k not in RESOURCES_KEYS:
                raise ValidationError(f'unknown key in jobs[{i}].resources: {k}')

        if 'memory' in resources:
            memory = resources['memory']
            if not isinstance(memory, str):
                raise ValidationError(f'jobs[{i}].resources.memory is not str')
            if not MEMORY_REGEX.fullmatch(memory):
                raise ValidationError(f'jobs[{i}].resources.memory must match regex: {MEMORY_REGEXPAT}')

        if 'cpu' in resources:
            cpu = resources['cpu']
            if not isinstance(cpu, str):
                raise ValidationError(f'jobs[{i}].resources.cpu is not str')
            if not CPU_REGEX.fullmatch(cpu):
                raise ValidationError(f'jobs[{i}].resources.cpu must match regex: {CPU_REGEXPAT}')

    if 'secrets' in job:
        secrets = job['secrets']
        if not isinstance(secrets, list):
            raise ValidationError(f'jobs[{i}].secrets is not list')
        for j, secret in enumerate(secrets):
            if not isinstance(secret, dict):
                raise ValidationError(f'jobs[{i}].secrets[{j}] is not dict')
            for k in secret:
                if k not in SECRET_KEYS:
                    raise ValidationError(f'unknown key in jobs[{i}].secrets[{j}]: {k}')

                if 'namespace' not in secret:
                    raise ValidationError(f'no required key namespace in jobs[{i}].secrets[{j}]')
                namespace = secret['namespace']
                if not isinstance(namespace, str):
                    raise ValidationError(f'jobs[{i}].secrets[{j}].namespace is not str')
                if len(namespace) > 253:
                    raise ValidationError(f'length of jobs[{i}].secrets[{j}].namespace must be <= 253')
                if not K8S_NAME_REGEX.fullmatch(namespace):
                    raise ValidationError(f'jobs[{i}].secrets[{j}].namespace must match regex: {K8S_NAME_REGEXPAT}')

                if 'name' not in secret:
                    raise ValidationError(f'no required key name in jobs[{i}].secrets[{j}]')
                name = secret['name']
                if not isinstance(name, str):
                    raise ValidationError(f'jobs[{i}].secrets[{j}].name is not str')
                if len(name) > 253:
                    raise ValidationError(f'length of jobs[{i}].secrets[{j}].name must be <= 253')
                if not K8S_NAME_REGEX.fullmatch(name):
                    raise ValidationError(f'jobs[{i}].secrets[{j}].name must match regex: {K8S_NAME_REGEXPAT}')

                if 'mount_path' not in secret:
                    raise ValidationError(f'no required key mount_path in jobs[{i}].secrets[{j}]')
                if not isinstance(name, str):
                    raise ValidationError(f'jobs[{i}].secrets[{j}].mount_path is not str')

    if 'service_account' in job:
        service_account = job['service_account']
        if not isinstance(service_account, dict):
            raise ValidationError(f'jobs[{i}].service_account is not dict')

        if 'namespace' not in service_account:
            raise ValidationError(f'no required key namespace in jobs[{i}].service_account')
        namespace = service_account['namespace']
        if not isinstance(namespace, str):
            raise ValidationError(f'jobs[{i}].service_account.namespace is not str')
        if len(namespace) > 253:
            raise ValidationError(f'length of jobs[{i}].service_account.namespace must be <= 253')
        if not K8S_NAME_REGEX.fullmatch(namespace):
            raise ValidationError(f'jobs[{i}].service_account.namespace must match regex: {K8S_NAME_REGEXPAT}')

        if 'name' not in service_account:
            raise ValidationError(f'no required key name in jobs[{i}].service_account')
        name = service_account['name']
        if not isinstance(name, str):
            raise ValidationError(f'jobs[{i}].service_account.name not str')
        if len(name) > 253:
            raise ValidationError(f'length of jobs[{i}].service_account.name must be <= 253')
        if not K8S_NAME_REGEX.fullmatch(name):
            raise ValidationError(f'jobs[{i}].service_account.name must match regex: {K8S_NAME_REGEXPAT}')
