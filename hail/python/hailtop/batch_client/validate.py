import re

# rough schema (without requiredness, value validation):
# jobs_schema = [{
#   'command': [str]
#   'env': [{
#     'name': str,
#     'value': str
#   }],
#   'image': str,
#   'mount_docker_socket': bool,
#   'resoures': {
#     'memory': str,
#     'cpu': str
#   },
#   secrets: [{
#     'namespace': str,
#     'name': str,
#     'mount_path': str
#   }],
#   service_account_name: str
# }]

JOB_KEYS = {
    'command', 'env', 'image', 'mount_docker_socket', 'resources', 'secrets', 'service_account_name'
}

ENV_VAR_KEYS = {'name', 'value'}

SECRET_KEYS = {'namespace', 'name', 'mount_path'}

RESOURCES_KEYS = {'memory', 'cpu'}

K8S_NAME_REGEXPAT = r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*'
K8S_NAME_REGEX = re.compile(K8S_NAME_REGEXPAT)

MEMORY_REGEXPAT = r'+?(?:[0-9]*[.])?[0-9]+[KMGTP]?'
MEMORY_REGEX = re.compile(MEMORY_REGEXPAT)

CPU_REGEXPAT = r'+?(?:[0-9]*[.])?[0-9]+[u]?'
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
        if k not in JOB_SPEC_KEYS:
            raise ValidationError(f'unknown key in jobs[{i}]: {k}')

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
            if not isinstance(value, str):
                raise ValidationError(f'jobs[{i}].env[{j}].value is not str')

    if 'image' not in job:
        raise ValidationError(f'no required key image in jobs[{i}]')
    image = job['image']
    if not isinstance(image, str):
        raise ValidationError(f'jobs[{i}].image not str')
    # FIXME validate image
    # https://github.com/docker/distribution/blob/master/reference/regexp.go#L68

    if 'mount_docker_socket' not in job:
        raise ValidationError(f'no required key mount_docker_socket in jobs[{i}]')
    mount_docker_socket = job['mount_docker_socket']
    if not isinstance(mount_docker_socket, bool):
        raise ValidationError(f'jobs[{i}].mount_docker_socket not bool')

    if 'resources' in job:
        resources = job['resources']
        if not isinstance(resources, dict):
            raise ValidationError(f'jobs[{i}].resources is not dict')
        for k in resources:
            if k not in RESOURCES_KEYS:
                raise ValidationError(f'unknown key in jobs[{i}].resources: {k}')

        if 'memory' not in secret:
            raise ValidationError(f'no required key memory in jobs[{i}].resources')
        memory = secret['memory']
        if not isinstance(memory, str):
            raise ValidationError(f'jobs[{i}].resources.memory is not str')
        if not MEMORY_REGEX.fullmatch(memory):
            raise ValidationError(f'jobs[{i}].resources.memory must match regex: {MEMORY_REGEXPAT}')

        if 'cpu' not in secret:
            raise ValidationError(f'no required key cpu in jobs[{i}].resources')
        cpu = secret['cpu']
        if not isinstance(cpu, str):
            raise ValidationError(f'jobs[{i}].resources.cpu is not str')
        if not CPU_REGEX.fullmatch(memory):
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
                    raies ValidationError(f'length of jobs[{i}].secrets[{j}].namespace must be <= 253')
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

    if 'service_account_name' not in job:
        raise ValidationError(f'no required key service_account_name in jobs[{i}]')
    service_account_name = job['service_account_name']
    if not isinstance(service_account_name, str):
        raise ValidationError(f'jobs[{i}].service_account_name not str')
    if len(service_account_name) > 253:
        raise ValidationError(f'length of jobs[{i}].service_account_name must be <= 253')
    if not K8S_NAME_REGEX.fullmatch(service_account_name):
        raise ValidationError(f'jobs[{i}].service_account_name must match regex: {K8S_NAME_REGEXPAT}')


def job_spec_to_k8s_pod_spec(job_spec):
    volumes = []
    volume_mounts = []

    if job_spec.get('mount_docker_socket', False):
        volumes.append({
            'name': 'docker-sock-volume',
            'hostPath': {
                'path': '/var/run/docker.sock',
                'type': 'File'
            }
        })
        volume_mounts.append({
            'mountPath': '/var/run/docker.sock',
            'name': 'docker-sock-volume'
        })

    if secrets in job_spec:
        secrets = job_spec['secrets']
        for secret in secrets:
            volumes.append({
                'name': secret['name'],
                'secret': {
                    'secretName': secret['name']
                }
            })
            volume_mounts.append({
                'mountPath': secret['mount_path'],
                'name': secret['name'],
                'readOnly': True
            })

    container = {
        'image': job_spec['image'],
        'name': 'main',
        'command': job_spec['command'],
        'volumes': volumes
    }
    if 'env' in job_spec:
        container['env'] = job_spec['env']
    if 'resources' in job_spec:
        requests = {}
        limits = {}
        job_resources = job_spec['resources']
        if 'memory' in job_resources:
            memory = job_resources['memory']
            requests['memory'] = memory
            limits['memory'] = memory
        if 'cpu' in job_resources:
            cpu = job_resources['cpu']
            requests['cpu'] = cpu
            limits['cpu'] = cpu
        container['resources'] = {
            'requests': requests,
            'limits': limits
        }
    pod_spec = {
        'containers': [container],
        'restartPolicy': 'Never'
        'volumeMounts': volume_mounts
    }
    if 'service_account_name' in job_spec:
        pod_spec['serviceAccountName'] = job_spec['service_account_name']
    return pod_spec
