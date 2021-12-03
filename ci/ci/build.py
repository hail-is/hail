import abc
import json
import logging
from collections import defaultdict, Counter
from shlex import quote as shq
import yaml
import jinja2
from typing import Dict, List, Optional
from hailtop.utils import flatten
from .utils import generate_token
from .environment import (
    DOCKER_PREFIX,
    DOMAIN,
    CI_UTILS_IMAGE,
    BUILDKIT_IMAGE,
    DEFAULT_NAMESPACE,
    STORAGE_URI,
    CLOUD,
)
from .globals import is_test_deployment
from gear.cloud_config import get_global_config

log = logging.getLogger('ci')

pretty_print_log = "jq -Rr '. as $raw | try \
(fromjson | if .hail_log == 1 then \
    ([.severity, .asctime, .filename, .funcNameAndLine, .message, .exc_info] | @tsv) \
    else $raw end) \
catch $raw'"


def expand_value_from(value, config):
    if isinstance(value, str):
        return value

    assert isinstance(value, dict)
    path = value['valueFrom']
    path = path.split('.')
    v = config
    for field in path:
        v = v[field]
    return v


def get_namespace(value, config):
    assert isinstance(value, dict)

    path = value['valueFrom'].split('.')

    assert len(path) == 2
    assert path[1] == 'name'

    v = config[path[0]]
    assert v['kind'] == 'createNamespace'

    return v['name']


class Code(abc.ABC):
    @abc.abstractmethod
    def short_str(self):
        pass

    @abc.abstractmethod
    def config(self):
        pass

    @abc.abstractmethod
    def repo_dir(self):
        """Path to repository on the ci (locally)."""

    @abc.abstractmethod
    def checkout_script(self):
        """Bash script to checkout out the code in the current directory."""


class StepParameters:
    def __init__(self, code, scope, json, name_step):
        self.code = code
        self.scope = scope
        self.json = json
        self.name_step = name_step


class BuildConfigurationError(Exception):
    pass


class BuildConfiguration:
    def __init__(self, code, config_str, scope, *, requested_step_names=(), excluded_step_names=()):
        if len(excluded_step_names) > 0 and scope != 'dev':
            raise BuildConfigurationError('Excluding build steps is only permitted in a dev scope')

        config = yaml.safe_load(config_str)
        name_step: Dict[str, Optional[Step]] = {}
        self.steps: List[Step] = []

        if requested_step_names:
            log.info(f"Constructing build configuration with steps: {requested_step_names}")

        for step_config in config['steps']:
            step_params = StepParameters(code, scope, step_config, name_step)
            step = Step.from_json(step_params)
            if not step.run_if_requested or step.name in requested_step_names:
                self.steps.append(step)
                name_step[step.name] = step
            else:
                name_step[step.name] = None

        # transitively close requested_step_names over dependencies
        if requested_step_names:
            visited = set()

            def request(step: Step):
                if step not in visited and step.name not in excluded_step_names:
                    visited.add(step)
                    for s2 in step.deps:
                        request(s2)

            for step_name in requested_step_names:
                request(name_step[step_name])
            self.steps = [s for s in self.steps if s in visited]

    def build(self, batch, code, scope):
        assert scope in ('deploy', 'test', 'dev')

        for step in self.steps:
            if (step.scopes is None or scope in step.scopes) and (step.clouds is None or CLOUD in step.clouds):
                step.build(batch, code, scope)

        if scope == 'dev':
            return

        step_to_parent_steps = defaultdict(set)
        for step in self.steps:
            for dep in step.all_deps():
                step_to_parent_steps[dep].add(step)

        for step in self.steps:
            parent_jobs = flatten([parent_step.wrapped_job() for parent_step in step_to_parent_steps[step]])

            log.info(
                f"Cleanup {step.name} after running {[parent_step.name for parent_step in step_to_parent_steps[step]]}"
            )

            if (step.scopes is None or scope in step.scopes) and (step.clouds is None or CLOUD in step.clouds):
                step.cleanup(batch, scope, parent_jobs)


class Step(abc.ABC):
    def __init__(self, params):
        json = params.json

        self.name = json['name']
        if 'dependsOn' in json:
            duplicates = [name for name, count in Counter(json['dependsOn']).items() if count > 1]
            if duplicates:
                raise BuildConfigurationError(f'found duplicate dependencies of {self.name}: {duplicates}')
            self.deps = [params.name_step[d] for d in json['dependsOn'] if params.name_step[d]]
        else:
            self.deps = []
        self.scopes = json.get('scopes')
        self.clouds = json.get('clouds')
        self.run_if_requested = json.get('runIfRequested', False)

        self.token = generate_token()

    def input_config(self, code, scope):
        config = {}
        config['global'] = get_global_config()
        config['token'] = self.token
        config['deploy'] = scope == 'deploy'
        config['scope'] = scope
        config['code'] = code.config()
        if self.deps:
            for d in self.deps:
                config[d.name] = d.config(scope)
        return config

    def deps_parents(self):
        if not self.deps:
            return None
        return flatten([d.wrapped_job() for d in self.deps])

    def all_deps(self):
        visited = set([self])
        frontier = [self]

        while frontier:
            current = frontier.pop()
            for d in current.deps:
                if d not in visited:
                    visited.add(d)
                    frontier.append(d)
        return visited

    @staticmethod
    def from_json(params):
        kind = params.json['kind']
        if kind == 'buildImage':
            return BuildImage2Step.from_json(params)
        if kind == 'buildImage2':
            return BuildImage2Step.from_json(params)
        if kind == 'runImage':
            return RunImageStep.from_json(params)
        if kind == 'createNamespace':
            return CreateNamespaceStep.from_json(params)
        if kind == 'deploy':
            return DeployStep.from_json(params)
        if kind in ('createDatabase', 'createDatabase2'):
            return CreateDatabaseStep.from_json(params)
        raise BuildConfigurationError(f'unknown build step kind: {kind}')

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @abc.abstractmethod
    def build(self, batch, code, scope):
        pass

    @abc.abstractmethod
    def cleanup(self, batch, scope, parents):
        pass


class BuildImage2Step(Step):
    def __init__(
        self, params, dockerfile, context_path, publish_as, inputs, resources
    ):  # pylint: disable=unused-argument
        super().__init__(params)
        self.dockerfile = dockerfile
        self.context_path = context_path
        self.publish_as = publish_as
        self.inputs = inputs
        self.resources = resources
        self.extra_cache_repository = None
        if publish_as:
            self.extra_cache_repository = f'{DOCKER_PREFIX}/{self.publish_as}'
        if params.scope == 'deploy' and publish_as and not is_test_deployment:
            self.base_image = f'{DOCKER_PREFIX}/{self.publish_as}'
        else:
            self.base_image = f'{DOCKER_PREFIX}/ci-intermediate'
        self.image = f'{self.base_image}:{self.token}'
        if publish_as:
            self.cache_repository = f'{DOCKER_PREFIX}/{self.publish_as}:cache'
        else:
            self.cache_repository = f'{DOCKER_PREFIX}/ci-intermediate:cache'
        self.job = None

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return BuildImage2Step(
            params,
            json['dockerFile'],
            json.get('contextPath'),
            json.get('publishAs'),
            json.get('inputs'),
            json.get('resources'),
        )

    def config(self, scope):  # pylint: disable=unused-argument
        return {'token': self.token, 'image': self.image}

    def build(self, batch, code, scope):
        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{STORAGE_URI}/build/{batch.attributes["token"]}{i["from"]}', i["to"]))
        else:
            input_files = None

        config = self.input_config(code, scope)

        context = self.context_path
        if not context:
            context = '/io'

        if isinstance(self.dockerfile, dict):
            assert ['inline'] == list(self.dockerfile.keys())
            unrendered_dockerfile = f'/home/user/Dockerfile.in.{self.token}'
            create_inline_dockerfile_if_present = f'echo {shq(self.dockerfile["inline"])} > {unrendered_dockerfile};\n'
        else:
            assert isinstance(self.dockerfile, str)
            unrendered_dockerfile = self.dockerfile
            create_inline_dockerfile_if_present = ''

        script = f'''
set -ex

{create_inline_dockerfile_if_present}

time python3 \
     ~/jinja2_render.py \
     {shq(json.dumps(config))} \
     {unrendered_dockerfile} \
     /home/user/Dockerfile

set +x
/bin/sh /home/user/convert-cloud-credentials-to-docker-auth-config
set -x

export BUILDKITD_FLAGS='--oci-worker-no-process-sandbox --oci-worker-snapshotter=overlayfs'
export BUILDCTL_CONNECT_RETRIES_MAX=100 # https://github.com/moby/buildkit/issues/1423
buildctl-daemonless.sh \
     build \
     --frontend dockerfile.v0 \
     --local context={shq(context)} \
     --local dockerfile=/home/user \
     --output 'type=image,"name={shq(self.image)},{shq(self.cache_repository)}",push=true' \
     --export-cache type=inline \
     --import-cache type=registry,ref={shq(self.cache_repository)} \
     --trace=/home/user/trace
cat /home/user/trace
'''

        log.info(f'step {self.name}, script:\n{script}')

        docker_registry = DOCKER_PREFIX.split('/')[0]
        job_env = {'REGISTRY': docker_registry}
        if CLOUD == 'gcp':
            credentials_name = 'GOOGLE_APPLICATION_CREDENTIALS'
        else:
            assert CLOUD == 'azure'
            credentials_name = 'AZURE_APPLICATION_CREDENTIALS'
        credentials_secret = {
            'namespace': DEFAULT_NAMESPACE,
            'name': 'registry-push-credentials',
            'mount_path': '/secrets/registry-push-credentials',
        }
        job_env[credentials_name] = '/secrets/registry-push-credentials/credentials.json'

        self.job = batch.create_job(
            BUILDKIT_IMAGE,
            command=['/bin/sh', '-c', script],
            secrets=[credentials_secret],
            env=job_env,
            attributes={'name': self.name},
            resources=self.resources,
            input_files=input_files,
            parents=self.deps_parents(),
            network='private',
            unconfined=True,
        )

    def cleanup(self, batch, scope, parents):
        if scope == 'deploy' and self.publish_as and not is_test_deployment:
            return

        if CLOUD == 'azure':
            image = 'mcr.microsoft.com/azure-cli'
            assert self.image.startswith(DOCKER_PREFIX)
            image_name = self.image[len(f'{DOCKER_PREFIX}/') :]
            script = f'''
set -x
date

set +x
USERNAME=$(cat /secrets/registry-push-credentials/credentials.json | jq -j '.appId')
PASSWORD=$(cat /secrets/registry-push-credentials/credentials.json | jq -j '.password')
TENANT=$(cat /secrets/registry-push-credentials/credentials.json | jq -j '.tenant')
az login --service-principal -u $USERNAME -p $PASSWORD --tenant $TENANT
set -x

until az acr repository untag -n {shq(DOCKER_PREFIX)} --image {shq(image_name)} || ! az acr repository show -n {shq(DOCKER_PREFIX)} --image {shq(image_name)}
do
    echo 'failed, will sleep 2 and retry'
    sleep 2
done

date
true
'''
        else:
            assert CLOUD == 'gcp'
            image = CI_UTILS_IMAGE
            script = f'''
set -x
date

gcloud -q auth activate-service-account \
  --key-file=/secrets/registry-push-credentials/credentials.json

until gcloud -q container images untag {shq(self.image)} || ! gcloud -q container images describe {shq(self.image)}
do
    echo 'failed, will sleep 2 and retry'
    sleep 2
done

date
true
'''

        self.job = batch.create_job(
            image,
            command=['bash', '-c', script],
            attributes={'name': f'cleanup_{self.name}'},
            secrets=[
                {
                    'namespace': DEFAULT_NAMESPACE,
                    'name': 'registry-push-credentials',
                    'mount_path': '/secrets/registry-push-credentials',
                }
            ],
            parents=parents,
            always_run=True,
            network='private',
            timeout=5 * 60,
        )


class RunImageStep(Step):
    def __init__(
        self, params, image, script, inputs, outputs, port, resources, service_account, secrets, always_run, timeout
    ):  # pylint: disable=unused-argument
        super().__init__(params)
        self.image = expand_value_from(image, self.input_config(params.code, params.scope))
        self.script = script
        self.inputs = inputs
        self.outputs = outputs
        self.port = port
        self.resources = resources
        if service_account:
            self.service_account = {
                'name': service_account['name'],
                'namespace': get_namespace(service_account['namespace'], self.input_config(params.code, params.scope)),
            }
        else:
            self.service_account = None
        self.secrets = secrets
        self.always_run = always_run
        self.timeout = timeout
        self.job = None

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return RunImageStep(
            params,
            json['image'],
            json['script'],
            json.get('inputs'),
            json.get('outputs'),
            json.get('port'),
            json.get('resources'),
            json.get('serviceAccount'),
            json.get('secrets'),
            json.get('alwaysRun', False),
            json.get('timeout', 3600),
        )

    def config(self, scope):  # pylint: disable=unused-argument
        return {'token': self.token}

    def build(self, batch, code, scope):
        template = jinja2.Template(self.script, undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        rendered_script = template.render(**self.input_config(code, scope))

        log.info(f'step {self.name}, rendered script:\n{rendered_script}')

        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{STORAGE_URI}/build/{batch.attributes["token"]}{i["from"]}', i["to"]))
        else:
            input_files = None

        if self.outputs:
            output_files = []
            for o in self.outputs:
                output_files.append((o["from"], f'{STORAGE_URI}/build/{batch.attributes["token"]}{o["to"]}'))
        else:
            output_files = None

        secrets = []
        if self.secrets:
            for secret in self.secrets:
                namespace = get_namespace(secret['namespace'], self.input_config(code, scope))
                name = expand_value_from(secret['name'], self.input_config(code, scope))
                mount_path = secret['mountPath']
                secrets.append({'namespace': namespace, 'name': name, 'mount_path': mount_path})

        self.job = batch.create_job(
            self.image,
            command=['bash', '-c', rendered_script],
            port=self.port,
            resources=self.resources,
            attributes={'name': self.name},
            input_files=input_files,
            output_files=output_files,
            secrets=secrets,
            service_account=self.service_account,
            parents=self.deps_parents(),
            always_run=self.always_run,
            timeout=self.timeout,
            network='private',
        )

    def cleanup(self, batch, scope, parents):
        pass


class CreateNamespaceStep(Step):
    def __init__(self, params, namespace_name, admin_service_account, public, secrets):
        super().__init__(params)
        self.namespace_name = namespace_name
        if admin_service_account:
            self.admin_service_account = {
                'name': admin_service_account['name'],
                'namespace': get_namespace(
                    admin_service_account['namespace'], self.input_config(params.code, params.scope)
                ),
            }
        else:
            self.admin_service_account = None
        self.public = public
        self.secrets = secrets
        self.job = None

        if is_test_deployment:
            assert self.namespace_name == 'default'
            self._name = DEFAULT_NAMESPACE
            return

        if params.scope == 'deploy':
            self._name = namespace_name
        elif params.scope == 'test':
            self._name = f'{params.code.short_str()}-{namespace_name}-{self.token}'
        elif params.scope == 'dev':
            self._name = params.code.namespace
        else:
            raise BuildConfigurationError(f"{params.scope} is not a valid scope for creating namespace")

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return CreateNamespaceStep(
            params,
            json['namespaceName'],
            json.get('adminServiceAccount'),
            json.get('public', False),
            json.get('secrets'),
        )

    def config(self, scope):  # pylint: disable=unused-argument
        return {'token': self.token, 'kind': 'createNamespace', 'name': self._name}

    def build(self, batch, code, scope):  # pylint: disable=unused-argument
        if is_test_deployment:
            return

        config = ""
        if scope in ['deploy', 'test']:
            # FIXME label
            config = (
                config
                + f'''\
apiVersion: v1
kind: Namespace
metadata:
  name: {self._name}
  labels:
    for: test
---
'''
            )
        config = (
            config
            + f'''\
kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: {self.namespace_name}-admin
  namespace: {self._name}
rules:
- apiGroups: [""]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin
  namespace: {self._name}
---
kind: RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: admin-{self.namespace_name}-admin-binding
  namespace: {self._name}
subjects:
- kind: ServiceAccount
  name: admin
  namespace: {self._name}
roleRef:
  kind: Role
  name: {self.namespace_name}-admin
  apiGroup: ""
'''
        )

        if self.admin_service_account:
            admin_service_account_name = self.admin_service_account['name']
            admin_service_account_namespace = self.admin_service_account['namespace']
            config = (
                config
                + f'''\
---
kind: RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: {admin_service_account_name}-{self.namespace_name}-admin-binding
  namespace: {self._name}
subjects:
- kind: ServiceAccount
  name: {admin_service_account_name}
  namespace: {admin_service_account_namespace}
roleRef:
  kind: Role
  name: {self.namespace_name}-admin
  apiGroup: ""
'''
            )

        script = f'''
set -ex
date

echo {shq(config)} | kubectl apply -f -
'''

        if self.secrets and scope != 'deploy':
            if self.namespace_name == 'default':
                script += f'''
kubectl -n {self.namespace_name} get -o json secret global-config \
  | jq '{{apiVersion:"v1",kind:"Secret","type":"Opaque",metadata:{{name:"global-config",namespace:"{self._name}"}},data:(.data + {{default_namespace:("{self._name}" | @base64)}})}}' \
  | kubectl -n {self._name} apply -f -
'''

            for s in self.secrets:
                if isinstance(s, str):
                    script += f'''
kubectl -n {self.namespace_name} get -o json secret {s} | jq 'del(.metadata) | .metadata.name = "{s}"' | kubectl -n {self._name} apply -f -
'''
                else:
                    clouds = s.get('clouds')
                    if clouds is None or CLOUD in clouds:
                        script += f'''
kubectl -n {self.namespace_name} get -o json secret {s["name"]} | jq 'del(.metadata) | .metadata.name = "{s["name"]}"' | kubectl -n {self._name} apply -f -
'''

        script += '''
date
'''

        self.job = batch.create_job(
            CI_UTILS_IMAGE,
            command=['bash', '-c', script],
            attributes={'name': self.name},
            # FIXME configuration
            service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
            parents=self.deps_parents(),
            network='private',
        )

    def cleanup(self, batch, scope, parents):
        if scope in ['deploy', 'dev'] or is_test_deployment:
            return

        script = f'''
set -x
date

until kubectl delete namespace --ignore-not-found=true {self._name}
do
    echo 'failed, will sleep 2 and retry'
    sleep 2
done

date
true
'''

        self.job = batch.create_job(
            CI_UTILS_IMAGE,
            command=['bash', '-c', script],
            attributes={'name': f'cleanup_{self.name}'},
            service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
            parents=parents,
            always_run=True,
            network='private',
        )


class DeployStep(Step):
    def __init__(self, params, namespace, config_file, link, wait):  # pylint: disable=unused-argument
        super().__init__(params)
        self.namespace = get_namespace(namespace, self.input_config(params.code, params.scope))
        self.config_file = config_file
        self.link = link
        self.wait = wait
        self.job = None

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return DeployStep(
            params,
            json['namespace'],
            # FIXME config_file
            json['config'],
            json.get('link'),
            json.get('wait'),
        )

    def config(self, scope):  # pylint: disable=unused-argument
        return {'token': self.token}

    def build(self, batch, code, scope):
        with open(f'{code.repo_dir()}/{self.config_file}', 'r') as f:
            template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
            rendered_config = template.render(**self.input_config(code, scope))

        script = '''\
set -ex
date
'''

        if self.wait:
            for w in self.wait:
                if w['kind'] == 'Pod':
                    script += f'''\
kubectl -n {self.namespace} delete --ignore-not-found pod {w['name']}
'''
        script += f'''
echo {shq(rendered_config)} | kubectl -n {self.namespace} apply -f -
'''

        if self.wait:
            for w in self.wait:
                name = w['name']
                if w['kind'] == 'Deployment':
                    assert w['for'] == 'available', w['for']
                    # FIXME what if the cluster isn't big enough?
                    script += f'''
set +e
kubectl -n {self.namespace} rollout status --timeout=1h deployment {name} && \
  kubectl -n {self.namespace} wait --timeout=1h --for=condition=available deployment {name}
EC=$?
kubectl -n {self.namespace} get deployment -l app={name} -o yaml
kubectl -n {self.namespace} get pods -l app={name} -o yaml
kubectl -n {self.namespace} logs --tail=999999 -l app={name} --all-containers=true | {pretty_print_log}
set -e
(exit $EC)
'''
                elif w['kind'] == 'Service':
                    assert w['for'] == 'alive', w['for']
                    resource_type = w.get('resource_type', 'deployment').lower()
                    timeout = w.get('timeout', 60)
                    if resource_type == 'statefulset':
                        wait_cmd = f'kubectl -n {self.namespace} wait --timeout=1h --for=condition=ready pods --selector=app={name}'
                        get_cmd = f'kubectl -n {self.namespace} get statefulset -l app={name} -o yaml'
                    else:
                        assert resource_type == 'deployment'
                        wait_cmd = (
                            f'kubectl -n {self.namespace} wait --timeout=1h --for=condition=available deployment {name}'
                        )
                        get_cmd = f'kubectl -n {self.namespace} get deployment -l app={name} -o yaml'

                    script += f'''
set +e
kubectl -n {self.namespace} rollout status --timeout=1h {resource_type} {name} && \
  {wait_cmd}
EC=$?
{get_cmd}
kubectl -n {self.namespace} get pods -l app={name} -o yaml
kubectl -n {self.namespace} logs --tail=999999 -l app={name} --all-containers=true | {pretty_print_log}
set -e
(exit $EC)
'''
                else:
                    assert w['kind'] == 'Pod', w['kind']
                    assert w['for'] == 'completed', w['for']
                    timeout = w.get('timeout', 300)
                    script += f'''
set +e
kubectl -n {self.namespace} wait --timeout=1h pod --for=condition=podscheduled {name} \
  && python3 wait-for.py {timeout} {self.namespace} Pod {name}
EC=$?
kubectl -n {self.namespace} get pod {name} -o yaml | {pretty_print_log}
kubectl -n {self.namespace} logs --tail=999999 {name} --all-containers=true | {pretty_print_log}
set -e
(exit $EC)
'''

        script += '''
date
'''

        attrs = {'name': self.name}
        if self.link is not None:
            attrs['link'] = ','.join(self.link)
            attrs['domain'] = f'{self.namespace}.internal.{DOMAIN}'

        self.job = batch.create_job(
            CI_UTILS_IMAGE,
            command=['bash', '-c', script],
            attributes=attrs,
            # FIXME configuration
            service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
            parents=self.deps_parents(),
            network='private',
        )

    def cleanup(self, batch, scope, parents):  # pylint: disable=unused-argument
        if self.wait:
            script = ''
            for w in self.wait:
                name = w['name']
                if w['kind'] == 'Deployment':
                    script += f'kubectl -n {self.namespace} logs --tail=999999 -l app={name} --all-containers=true | {pretty_print_log}\n'
                elif w['kind'] == 'Service':
                    assert w['for'] == 'alive', w['for']
                    script += f'kubectl -n {self.namespace} logs --tail=999999 -l app={name} --all-containers=true | {pretty_print_log}\n'
                else:
                    assert w['kind'] == 'Pod', w['kind']
                    script += f'kubectl -n {self.namespace} logs --tail=999999 {name} --all-containers=true | {pretty_print_log}\n'
            script += 'date\n'
            self.job = batch.create_job(
                CI_UTILS_IMAGE,
                command=['bash', '-c', script],
                attributes={'name': self.name + '_logs'},
                # FIXME configuration
                service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
                parents=parents,
                always_run=True,
                network='private',
            )


class CreateDatabaseStep(Step):
    def __init__(self, params, database_name, namespace, migrations, shutdowns, inputs):
        super().__init__(params)

        config = self.input_config(params.code, params.scope)

        # FIXME validate
        self.database_name = database_name
        self.namespace = get_namespace(namespace, config)
        self.migrations = migrations

        for s in shutdowns:
            s['namespace'] = get_namespace(s['namespace'], config)
        self.shutdowns = shutdowns

        self.inputs = inputs
        self.create_passwords_job = None
        self.create_database_job = None
        self.cleanup_job = None

        if params.scope == 'dev':
            self.database_server_config_namespace = params.code.namespace
        else:
            self.database_server_config_namespace = DEFAULT_NAMESPACE

        self.cant_create_database = is_test_deployment or params.scope == 'dev'

        # MySQL user name can be up to 16 characters long before MySQL 5.7.8 (32 after)
        if self.cant_create_database:
            self._name = None
            self.admin_username = None
            self.user_username = None
        elif params.scope == 'deploy':
            self._name = database_name
            self.admin_username = f'{database_name}-admin'
            self.user_username = f'{database_name}-user'
        else:
            assert params.scope == 'test'
            self._name = f'{params.code.short_str()}-{database_name}-{self.token}'
            self.admin_username = generate_token()
            self.user_username = generate_token()

        self.admin_password_file = f'/io/{self.admin_username}.pwd'
        self.user_password_file = f'/io/{self.user_username}.pwd'

        self.admin_secret_name = f'sql-{self.database_name}-admin-config'
        self.user_secret_name = f'sql-{self.database_name}-user-config'

    def wrapped_job(self):
        if self.cleanup_job:
            return [self.cleanup_job]
        if self.create_passwords_job:
            assert self.create_database_job is not None
            return [self.create_passwords_job, self.create_database_job]
        if self.create_database_job:
            return [self.create_database_job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return CreateDatabaseStep(
            params,
            json['databaseName'],
            json['namespace'],
            json['migrations'],
            json.get('shutdowns', []),
            json.get('inputs'),
        )

    def config(self, scope):  # pylint: disable=unused-argument
        return {
            'token': self.token,
            'admin_secret_name': self.admin_secret_name,
            'user_secret_name': self.user_secret_name,
        }

    def build(self, batch, code, scope):  # pylint: disable=unused-argument
        create_database_config = {
            'cloud': CLOUD,
            'namespace': self.namespace,
            'scope': scope,
            'database_name': self.database_name,
            '_name': self._name,
            'admin_username': self.admin_username,
            'user_username': self.user_username,
            'admin_password_file': self.admin_password_file,
            'user_password_file': self.user_password_file,
            'cant_create_database': self.cant_create_database,
            'migrations': self.migrations,
            'shutdowns': self.shutdowns,
        }

        create_passwords_script = f'''
set -ex

LC_ALL=C tr -dc '[:alnum:]' </dev/urandom | head -c 16 > {self.admin_password_file}
LC_ALL=C tr -dc '[:alnum:]' </dev/urandom | head -c 16 > {self.user_password_file}
'''

        create_database_script = f'''
set -ex

create_database_config={shq(json.dumps(create_database_config, indent=2))}
python3 create_database.py <<EOF
$create_database_config
EOF
'''

        input_files = []
        if self.inputs:
            for i in self.inputs:
                input_files.append((f'{STORAGE_URI}/build/{batch.attributes["token"]}{i["from"]}', i["to"]))

        if not self.cant_create_database:
            password_files_input = [
                (
                    f'{STORAGE_URI}/build/{batch.attributes["token"]}/{self.admin_password_file}',
                    self.admin_password_file,
                ),
                (f'{STORAGE_URI}/build/{batch.attributes["token"]}/{self.user_password_file}', self.user_password_file),
            ]
            input_files.extend(password_files_input)

            self.create_passwords_job = batch.create_job(
                CI_UTILS_IMAGE,
                command=['bash', '-c', create_passwords_script],
                attributes={'name': self.name + "_create_passwords"},
                output_files=[(x[1], x[0]) for x in password_files_input],
                parents=self.deps_parents(),
            )

        self.create_database_job = batch.create_job(
            CI_UTILS_IMAGE,
            command=['bash', '-c', create_database_script],
            attributes={'name': self.name},
            secrets=[
                {
                    'namespace': self.database_server_config_namespace,
                    'name': 'database-server-config',
                    'mount_path': '/sql-config',
                }
            ],
            service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
            input_files=input_files,
            parents=[self.create_passwords_job] if self.create_passwords_job else self.deps_parents(),
            network='private',
        )

    def cleanup(self, batch, scope, parents):
        if scope in ['deploy', 'dev'] or self.cant_create_database:
            return

        cleanup_script = f'''
set -ex

commands=$(mktemp)

cat >$commands <<EOF
DROP DATABASE IF EXISTS \\`{self._name}\\`;
DROP USER IF EXISTS '{self.admin_username}';
DROP USER IF EXISTS '{self.user_username}';
EOF

until mysql --defaults-extra-file=/sql-config/sql-config.cnf <$commands
do
    echo 'failed, will sleep 2 and retry'
    sleep 2
done

'''

        self.cleanup_job = batch.create_job(
            CI_UTILS_IMAGE,
            command=['bash', '-c', cleanup_script],
            attributes={'name': f'cleanup_{self.name}'},
            secrets=[
                {
                    'namespace': self.database_server_config_namespace,
                    'name': 'database-server-config',
                    'mount_path': '/sql-config',
                }
            ],
            service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
            parents=parents,
            always_run=True,
            network='private',
        )
