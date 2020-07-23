import re

from hailtop.batch_client.parse import (MEMORY_REGEX, MEMORY_REGEXPAT,
                                        CPU_REGEX, CPU_REGEXPAT)

# rough schema (without requiredness, value validation):
# jobs_schema = [{
#   'always_run': bool,
#   'attributes': {str: str},
#   'command': [str],
#   'env': [{
#     'name': str,
#     'value': str
#   }],
#   'gcsfuse': [{"bucket": str, "mount_path": str, "read_only": bool}],
#   'image': str,
#   'input_files': [{"from": str, "to": str}],
#   'job_id': int,
#   'mount_docker_socket': bool,
#   'output_files': [{"from": str, "to": str}],
#   'parent_ids': [int],
#   'port': int,
#   'pvc_size': str,
#   'requester_pays_project': str,
#   'resoures': {
#     'memory': str,
#     'cpu': str
#   },
#   'secrets': [{
#     'namespace': str,
#     'name': str,
#     'mount_path': str
#   }],
#   'service_account': {
#     'namespace': str,
#     'name': str
#   },
#   'timeout': float or int
# }]

JOB_KEYS = {
    'always_run', 'attributes', 'command', 'env', 'gcsfuse', 'image', 'input_files', 'job_id', 'mount_docker_socket', 'output_files', 'parent_ids', 'pvc_size', 'port', 'requester_pays_project', 'resources', 'secrets', 'service_account', 'timeout'
}

ENV_VAR_KEYS = {'name', 'value'}

SECRET_KEYS = {'namespace', 'name', 'mount_path'}

RESOURCES_KEYS = {'memory', 'cpu'}

FILE_KEYS = {'from', 'to'}

GCSFUSE_KEYS = {'bucket', 'mount_path', 'read_only'}

K8S_NAME_REGEXPAT = r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*'
K8S_NAME_REGEX = re.compile(K8S_NAME_REGEXPAT)


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

    if 'gcsfuse' in job:
        gcsfuse = job['gcsfuse']
        if not isinstance(gcsfuse, list):
            raise ValidationError(f'jobs[{i}].gcsfuse not list')

        for j, b in enumerate(gcsfuse):
            if not isinstance(b, dict):
                raise ValidationError(f'jobs[{i}].gcsfuse[{j}] not dict')

            for k in b:
                if k not in GCSFUSE_KEYS:
                    raise ValidationError(f'unknown key in jobs[{i}].gcsfuse[{j}]: {k}')

            if 'bucket' not in b:
                raise ValidationError(f'no required key bucket in jobs[{i}].gcsfuse[{j}]')
            bucket = b['bucket']
            if not isinstance(bucket, str):
                raise ValidationError(f'jobs[{i}].gcsfuse[{j}].bucket is not str')

            if 'mount_path' not in b:
                raise ValidationError(f'no required key mount_path in jobs[{i}].gcsfuse[{j}]')
            mount_path = b['mount_path']
            if not isinstance(mount_path, str):
                raise ValidationError(f'jobs[{i}].gcsfuse[{j}].mount_path is not str')

            if 'read_only' not in b:
                raise ValidationError(f'no required key read_only in jobs[{i}].gcsfuse[{j}]')
            read_only = b['read_only']
            if not isinstance(read_only, bool):
                raise ValidationError(f'jobs[{i}].gcsfuse[{j}].read_only is not bool')

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

    if 'port' in job:
        port = job['port']
        if not isinstance(port, int):
            raise ValidationError(f'jobs[{i}].port not int')

    if 'pvc_size' in job:
        pvc_size = job['pvc_size']
        if not isinstance(pvc_size, str):
            raise ValidationError(f'jobs[{i}].pvc_size not str')
        if not MEMORY_REGEX.fullmatch(pvc_size):
            raise ValidationError(f'jobs[{i}].pvc_size must match regex: {MEMORY_REGEXPAT}')

    if 'requester_pays_project' in job:
        requester_pays_project = job['requester_pays_project']
        if not isinstance(requester_pays_project, str):
            raise ValidationError(f'jobs[{i}].requester_pays_project not str')

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

    if 'timeout' in job:
        timeout = job['timeout']
        if not isinstance(timeout, float) and not isinstance(timeout, int):
            raise ValidationError(f'jobs[{i}].timeout not numeric')
        if timeout < 0:
            raise ValidationError(f'jobs[{i}].timeout is not a positive number')


# rough schema
# batch_schema = {
#   'attributes': {str: str},
#   'billing_project': str,
#   'callback': str,
#   'n_jobs': int,
#   'token': str
# }

BATCH_KEYS = {
    'attributes', 'billing_project', 'callback', 'n_jobs', 'token'
}


def validate_batch(batch):
    if not isinstance(batch, dict):
        raise ValidationError(f'batch not dict')

    for k in batch:
        if k not in BATCH_KEYS:
            raise ValidationError(f'unknown key in batch: {k}')

    attributes = batch.get('attributes')
    if attributes is not None:
        if not isinstance(attributes, dict):
            raise ValidationError(f'batch.attributes is not dict')
        for k, v in attributes.items():
            if not isinstance(k, str):
                raise ValidationError(f'batch.attributes has non-str key')
            if not isinstance(v, str):
                raise ValidationError(f'batch.attributes has non-str value')

    if 'billing_project' not in batch:
        raise ValidationError('no required key billing_project in batch')
    billing_project = batch['billing_project']
    if not isinstance(billing_project, str):
        raise ValidationError('batch.billing_project is not str')

    callback = batch.get('callback')
    if callback is not None:
        if not isinstance(callback, str):
            raise ValidationError(f'batch.callback not str')

    if 'n_jobs' not in batch:
        raise ValidationError('no required key n_jobs in batch')
    n_jobs = batch['n_jobs']
    if not isinstance(n_jobs, int):
        raise ValidationError('batch.n_jobs is not int')

    if 'token' not in batch:
        raise ValidationError('no required key token in batch')
    token = batch['token']
    if not isinstance(token, str):
        raise ValidationError('batch.token is not str')
