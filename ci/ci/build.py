import abc
import json
import logging
from collections import Counter, defaultdict
from shlex import quote as shq
from typing import Dict, List, Optional, Sequence, Set, TypedDict

import jinja2
import yaml

from gear.cloud_config import get_global_config
from hailtop.utils import RETRY_FUNCTION_SCRIPT, flatten

from .environment import (
    BUILDKIT_IMAGE,
    CI_UTILS_IMAGE,
    CLOUD,
    DEFAULT_NAMESPACE,
    DOCKER_PREFIX,
    DOMAIN,
    REGION,
    STORAGE_URI,
)
from .globals import is_test_deployment
from .utils import generate_token

log = logging.getLogger('ci')

pretty_print_log = "jq -Rr '. as $raw | try \
(fromjson | if .hail_log == 1 then \
    ([.severity, .asctime, .filename, .funcNameAndLine, .message, .exc_info] | @tsv) \
    else $raw end) \
catch $raw'"


class ServiceAccount(TypedDict):
    name: str
    namespace: str


def expand_value_from(value, config) -> str:
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
    def short_str(self) -> str:
        pass

    @abc.abstractmethod
    def config(self) -> Dict[str, str]:
        pass

    @abc.abstractmethod
    def repo_dir(self) -> str:
        """Path to repository on the ci (locally)."""

    @abc.abstractmethod
    def checkout_script(self) -> str:
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
    def __init__(
        self,
        code: Code,
        config_str: str,
        scope: str,
        *,
        requested_step_names: Sequence[str] = (),
        excluded_step_names: Sequence[str] = (),
    ):
        if len(excluded_step_names) > 0 and scope != 'dev':
            raise BuildConfigurationError('Excluding build steps is only permitted in a dev scope')

        config = yaml.safe_load(config_str)
        if requested_step_names:
            log.info(f"Constructing build configuration with steps: {requested_step_names}")

        runnable_steps: List[Step] = []
        name_step: Dict[str, Step] = {}
        for step_config in config['steps']:
            step = Step.from_json(StepParameters(code, scope, step_config, name_step))
            if step.name not in excluded_step_names and step.can_run_in_current_cloud():
                name_step[step.name] = step
                runnable_steps.append(step)

        if requested_step_names:
            # transitively close requested_step_names over dependencies
            visited = set()

            def visit_dependent(step: Step):
                if step not in visited and step.name not in excluded_step_names:
                    if not step.can_run_in_current_cloud():
                        raise BuildConfigurationError(f'Step {step.name} cannot be run in cloud {CLOUD}')
                    visited.add(step)
                    for s2 in step.deps:
                        if not s2.run_if_requested:
                            visit_dependent(s2)

            for step_name in requested_step_names:
                visit_dependent(name_step[step_name])
            self.steps = [step for step in runnable_steps if step in visited]
        else:
            self.steps = [step for step in runnable_steps if not step.run_if_requested]

    def build(self, batch, code, scope):
        assert scope in ('deploy', 'test', 'dev')

        for step in self.steps:
            if step.can_run_in_scope(scope):
                assert step.can_run_in_current_cloud()
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

            if step.can_run_in_scope(scope):
                step.cleanup(batch, scope, parent_jobs)

    def namespace(self) -> Optional[str]:
        # build.yaml allows for multiple namespaces, but
        # in actuality we only ever use 1 and make many assumptions
        # around there being a 1:1 correspondence between builds and namespaces
        namespaces = {s.namespace for s in self.steps if isinstance(s, DeployStep)}
        assert len(namespaces) <= 1
        return namespaces.pop() if len(namespaces) == 1 else None

    def deployed_services(self) -> List[str]:
        services = []
        for s in self.steps:
            if isinstance(s, DeployStep):
                services.extend(s.services())
        return services


class Step(abc.ABC):
    def __init__(self, params):
        json = params.json

        self.name = json['name']
        self.deps: List[Step] = []
        if 'dependsOn' in json:
            duplicates = [name for name, count in Counter(json['dependsOn']).items() if count > 1]
            if duplicates:
                raise BuildConfigurationError(f'found duplicate dependencies of {self.name}: {duplicates}')
            self.deps = [params.name_step[d] for d in json['dependsOn'] if d in params.name_step]

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
        visited: Set[Step] = set([self])
        frontier: List[Step] = [self]

        while frontier:
            current = frontier.pop()
            for d in current.deps:
                if d not in visited:
                    visited.add(d)
                    frontier.append(d)
        return visited

    def can_run_in_current_cloud(self):
        return self.clouds is None or CLOUD in self.clouds

    def can_run_in_scope(self, scope: str):
        return self.scopes is None or scope in self.scopes

    @staticmethod
    def from_json(params: StepParameters):
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
        if kind == 'createDatabase2':
            return CreateDatabase2Step.from_json(params)
        raise BuildConfigurationError(f'unknown build step kind: {kind}')

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @abc.abstractmethod
    def wrapped_job(self) -> list:
        pass

    @abc.abstractmethod
    def build(self, batch, code, scope):
        pass

    @abc.abstractmethod
    def config(self, scope) -> dict:
        pass

    @abc.abstractmethod
    def cleanup(self, batch, scope, parents):
        pass


class BuildImage2Step(Step):
    def __init__(
        self,
        params: StepParameters,
        dockerfile,
        context_path,
        publish_as,
        inputs,
        resources,
        build_args,
    ):  # pylint: disable=unused-argument
        super().__init__(params)
        self.dockerfile = dockerfile
        self.context_path = context_path
        self.inputs = inputs
        self.resources = resources

        self.build_args = {
            arg['name']: expand_value_from(arg['value'], self.input_config(params.code, params.scope))
            for arg in build_args
        }

        image_name = publish_as
        self.base_image = f'{DOCKER_PREFIX}/{image_name}'
        self.main_branch_cache_repository = f'{self.base_image}:cache'

        if params.scope == 'deploy':
            if is_test_deployment:
                # CIs that don't live in default doing a deploy
                # should not clobber the main `cache` tag
                self.cache_repository = f'{self.base_image}:cache-{DEFAULT_NAMESPACE}-deploy'
                self.image = f'{self.base_image}:test-deploy-{self.token}'
            else:
                self.cache_repository = self.main_branch_cache_repository
                self.image = f'{self.base_image}:deploy-{self.token}'
        elif params.scope == 'dev':
            dev_user = params.code.config()['user']
            self.cache_repository = f'{self.base_image}:cache-{dev_user}'
            self.image = f'{self.base_image}:dev-{self.token}'
        else:
            assert params.scope == 'test'
            pr_number = params.code.config()['number']
            self.cache_repository = f'{self.base_image}:cache-pr-{pr_number}'
            self.image = f'{self.base_image}:test-pr-{pr_number}-{self.token}'

        self.job = None

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params: StepParameters):
        json = params.json
        return BuildImage2Step(
            params,
            json['dockerFile'],
            json.get('contextPath'),
            json['publishAs'],
            json.get('inputs'),
            json.get('resources'),
            json.get('buildArgs', []),
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

        build_args_str = ' \\\n'.join(
            f'--opt build-arg:{shq(name)}={shq(value)}' for name, value in self.build_args.items()
        )

        if isinstance(self.dockerfile, dict):
            assert ['inline'] == list(self.dockerfile.keys())
            unrendered_dockerfile = f'/home/user/Dockerfile.in.{self.token}'
            create_inline_dockerfile_if_present = f'echo {shq(self.dockerfile["inline"])} > {unrendered_dockerfile};\n'
        else:
            assert isinstance(self.dockerfile, str)
            unrendered_dockerfile = self.dockerfile
            create_inline_dockerfile_if_present = ''

        script = f"""
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

{RETRY_FUNCTION_SCRIPT}

export BUILDKITD_FLAGS='--oci-worker-no-process-sandbox --oci-worker-snapshotter=overlayfs'
export BUILDCTL_CONNECT_RETRIES_MAX=100 # https://github.com/moby/buildkit/issues/1423
retry buildctl-daemonless.sh \\
     build \\
     --frontend dockerfile.v0 \\
     --local context={shq(context)} \\
     --local dockerfile=/home/user \\
     --output 'type=image,"name={shq(self.image)},{shq(self.cache_repository)}",push=true' \\
     --export-cache type=inline \\
     --import-cache type=registry,ref={shq(self.cache_repository)} \\
     --import-cache type=registry,ref={shq(self.main_branch_cache_repository)} \\
     {build_args_str} \\
     --trace=/home/user/trace
cat /home/user/trace
"""

        log.info(f'step {self.name}, script:\n{script}')

        docker_registry = DOCKER_PREFIX.split('/')[0]
        job_env = {'REGISTRY': docker_registry}

        self.job = batch.create_job(
            BUILDKIT_IMAGE,
            command=['/bin/sh', '-c', script],
            env=job_env,
            attributes={'name': self.name},
            resources=self.resources,
            input_files=input_files,
            parents=self.deps_parents(),
            network='private',
            unconfined=True,
            regions=[REGION],
        )

    def cleanup(self, batch, scope, parents):
        if scope == 'deploy' and not is_test_deployment:
            return

        if CLOUD == 'azure':
            image = 'mcr.microsoft.com/azure-cli'
            assert self.image.startswith(DOCKER_PREFIX + '/')
            image_name = self.image.removeprefix(DOCKER_PREFIX + '/')
            script = f"""
set -x
date

set +x
USERNAME=$(cat $AZURE_APPLICATION_CREDENTIALS | jq -j '.appId')
PASSWORD=$(cat $AZURE_APPLICATION_CREDENTIALS | jq -j '.password')
TENANT=$(cat $AZURE_APPLICATION_CREDENTIALS | jq -j '.tenant')
az login --service-principal -u $USERNAME -p $PASSWORD --tenant $TENANT
set -x

until az acr repository untag -n {shq(DOCKER_PREFIX)} --image {shq(image_name)} || ! az acr repository show -n {shq(DOCKER_PREFIX)} --image {shq(image_name)}
do
    echo 'failed, will sleep 2 and retry'
    sleep 2
done

date
true
"""
        else:
            assert CLOUD == 'gcp'
            image = CI_UTILS_IMAGE
            script = f"""
set -x
date

until gcloud -q container images untag {shq(self.image)} || ! gcloud -q container images describe {shq(self.image)}
do
    echo 'failed, will sleep 2 and retry'
    sleep 2
done

date
true
"""

        self.job = batch.create_job(
            image,
            command=['bash', '-c', script],
            attributes={'name': f'cleanup_{self.name}'},
            resources={'cpu': '0.25'},
            parents=parents,
            always_run=True,
            network='private',
            timeout=5 * 60,
            regions=[REGION],
        )


class RunImageStep(Step):
    def __init__(
        self,
        params,
        image,
        script,
        inputs,
        outputs,
        port,
        resources,
        service_account,
        secrets,
        always_run,
        timeout,
        num_splits,
    ):  # pylint: disable=unused-argument
        super().__init__(params)
        self.image = expand_value_from(image, self.input_config(params.code, params.scope))
        self.script = script
        self.inputs = inputs
        self.outputs = outputs
        self.port = port
        self.resources = resources
        self.service_account: Optional[ServiceAccount]
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
        self.jobs = []
        self.num_splits = num_splits

    def wrapped_job(self):
        return self.jobs

    @staticmethod
    def from_json(params: StepParameters):
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
            json.get('numSplits', 1),
        )

    def config(self, scope):  # pylint: disable=unused-argument
        return {'token': self.token}

    def build(self, batch, code, scope):
        if self.num_splits == 1:
            self.jobs = [self._build_job(batch, code, scope, self.name, None, None)]
        else:
            self.jobs = [
                self._build_job(
                    batch,
                    code,
                    scope,
                    f'{self.name}_{i}',
                    {'HAIL_RUN_IMAGE_SPLITS': str(self.num_splits), 'HAIL_RUN_IMAGE_SPLIT_INDEX': str(i)},
                    f'/{self.name}_{i}',
                )
                for i in range(self.num_splits)
            ]

    def _build_job(self, batch, code, scope, job_name, env, output_prefix):
        template = jinja2.Template(self.script, undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        rendered_script = template.render(**self.input_config(code, scope))

        log.info(f'step {job_name}, rendered script:\n{rendered_script}')

        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{STORAGE_URI}/build/{batch.attributes["token"]}{i["from"]}', i["to"]))
        else:
            input_files = None

        if self.outputs:
            output_files = []
            for o in self.outputs:
                prefixed_path = o["to"] if output_prefix is None else output_prefix + o["to"]
                output_files.append((o["from"], f'{STORAGE_URI}/build/{batch.attributes["token"]}{prefixed_path}'))
        else:
            output_files = None

        secrets = []
        if self.secrets:
            for secret in self.secrets:
                namespace = get_namespace(secret['namespace'], self.input_config(code, scope))
                name = expand_value_from(secret['name'], self.input_config(code, scope))
                mount_path = secret['mountPath']
                secrets.append({'namespace': namespace, 'name': name, 'mount_path': mount_path})

        return batch.create_job(
            self.image,
            command=['bash', '-c', rendered_script],
            port=self.port,
            resources=self.resources,
            attributes={'name': job_name},
            input_files=input_files,
            output_files=output_files,
            secrets=secrets,
            service_account=self.service_account,
            parents=self.deps_parents(),
            always_run=self.always_run,
            timeout=self.timeout,
            network='private',
            env=env,
            regions=[REGION],
        )

    def cleanup(self, batch, scope, parents):
        pass


class CreateNamespaceStep(Step):
    def __init__(self, params, namespace_name, secrets):
        super().__init__(params)
        self.namespace_name = namespace_name
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
    def from_json(params: StepParameters):
        json = params.json
        return CreateNamespaceStep(
            params,
            json['namespaceName'],
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
                + f"""\
apiVersion: v1
kind: Namespace
metadata:
  name: {self._name}
  labels:
    for: test
---
"""
            )
        config = (
            config
            + f"""\
kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: {self.namespace_name}-admin
  namespace: {self._name}
rules:
- apiGroups: [""]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["apps"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin
  namespace: {self._name}
---
apiVersion: v1
kind: Secret
type: kubernetes.io/service-account-token
metadata:
  name: admin-token
  namespace: {self._name}
  annotations:
    kubernetes.io/service-account.name: admin
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
"""
        )

        script = f"""
set -ex
date

echo {shq(config)} | kubectl apply -f -
"""

        if self.secrets and scope != 'deploy':
            if self.namespace_name == 'default':
                script += f"""
kubectl -n {self.namespace_name} get -o json secret global-config \
  | jq '{{apiVersion:"v1",kind:"Secret","type":"Opaque",metadata:{{name:"global-config",namespace:"{self._name}"}},data:(.data + {{default_namespace:("{self._name}" | @base64)}})}}' \
  | kubectl -n {self._name} apply -f -
"""

            for s in self.secrets:
                name = s['name']
                if s.get('clouds') is None or CLOUD in s['clouds']:
                    script += f"""
kubectl -n {self.namespace_name} get -o json secret {name} | jq 'del(.metadata) | .metadata.name = "{name}"' | kubectl -n {self._name} apply -f - { '|| true' if s.get('optional') is True else ''}
"""

        script += """
date
"""

        self.job = batch.create_job(
            CI_UTILS_IMAGE,
            command=['bash', '-c', script],
            attributes={'name': self.name},
            # FIXME configuration
            service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
            parents=self.deps_parents(),
            network='private',
            regions=[REGION],
        )

    def cleanup(self, batch, scope, parents):
        if scope in ['deploy', 'dev'] or is_test_deployment:
            return

        script = f"""
set -x
date

until kubectl delete namespace --ignore-not-found=true {self._name}
do
    echo 'failed, will sleep 2 and retry'
    sleep 2
done

date
true
"""

        self.job = batch.create_job(
            CI_UTILS_IMAGE,
            command=['bash', '-c', script],
            attributes={'name': f'cleanup_{self.name}'},
            service_account={'namespace': DEFAULT_NAMESPACE, 'name': 'ci-agent'},
            parents=parents,
            always_run=True,
            network='private',
            regions=[REGION],
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
    def from_json(params: StepParameters):
        json = params.json
        return DeployStep(
            params,
            json['namespace'],
            # FIXME config_file
            json['config'],
            json.get('link'),
            json.get('wait'),
        )

    def services(self):
        if self.wait:
            return [w['name'] for w in self.wait]
        return []

    def config(self, scope):  # pylint: disable=unused-argument
        return {'token': self.token}

    def build(self, batch, code, scope):
        with open(f'{code.repo_dir()}/{self.config_file}', 'r', encoding='utf-8') as f:
            template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
            rendered_config = template.render(**self.input_config(code, scope))

        script = """\
set -ex
date
"""

        if self.wait:
            for w in self.wait:
                if w['kind'] == 'Pod':
                    script += f"""\
kubectl -n {self.namespace} delete --ignore-not-found pod {w['name']}
"""
        script += f"""
echo {shq(rendered_config)} | kubectl -n {self.namespace} apply -f -
"""

        if self.wait:
            for w in self.wait:
                name = w['name']
                if w['kind'] == 'Deployment':
                    assert w['for'] == 'available', w['for']
                    # FIXME what if the cluster isn't big enough?
                    script += f"""
set +e
kubectl -n {self.namespace} rollout status --timeout=1h deployment {name} && \
  kubectl -n {self.namespace} wait --timeout=1h --for=condition=available deployment {name}
EC=$?
kubectl -n {self.namespace} get deployment -l app={name} -o yaml
kubectl -n {self.namespace} get pods -l app={name} -o yaml
kubectl -n {self.namespace} logs --tail=999999 -l app={name} --all-containers=true | {pretty_print_log}
set -e
(exit $EC)
"""
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

                    script += f"""
set +e
kubectl -n {self.namespace} rollout status --timeout=1h {resource_type} {name} && \
  {wait_cmd}
EC=$?
{get_cmd}
kubectl -n {self.namespace} get pods -l app={name} -o yaml
kubectl -n {self.namespace} logs --tail=999999 -l app={name} --all-containers=true | {pretty_print_log}
set -e
(exit $EC)
"""
                else:
                    assert w['kind'] == 'Pod', w['kind']
                    assert w['for'] == 'completed', w['for']
                    timeout = w.get('timeout', 300)
                    script += f"""
set +e
kubectl -n {self.namespace} wait --timeout=1h pod --for=condition=podscheduled {name} \
  && python3 wait-for.py {timeout} {self.namespace} Pod {name}
EC=$?
kubectl -n {self.namespace} get pod {name} -o yaml | {pretty_print_log}
kubectl -n {self.namespace} logs --tail=999999 {name} --all-containers=true | {pretty_print_log}
set -e
(exit $EC)
"""

        script += """
date
"""

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
            resources={'cpu': '0.25'},
            parents=self.deps_parents(),
            network='private',
            regions=[REGION],
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
                resources={'cpu': '0.25'},
                parents=parents,
                always_run=True,
                network='private',
                regions=[REGION],
            )


class CreateDatabase2Step(Step):
    def __init__(self, params, database_name, namespace, migrations, shutdowns, inputs, image):
        super().__init__(params)

        config = self.input_config(params.code, params.scope)

        self.image = expand_value_from(image, self.input_config(params.code, params.scope))

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

        self.cant_create_database = False

        # MySQL user name can be up to 16 characters long before MySQL 5.7.8 (32 after)
        if params.scope == 'deploy':
            self._name = database_name
            self.admin_username = f'{database_name}-admin'
            self.user_username = f'{database_name}-user'
        elif params.scope == 'dev':
            dev_username = params.code.config()['user']
            self._name = f'{dev_username}-{database_name}'
            self.admin_username = f'{dev_username}-{database_name}-admin'
            self.user_username = f'{dev_username}-{database_name}-user'
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
    def from_json(params: StepParameters):
        json = params.json
        return CreateDatabase2Step(
            params,
            json['databaseName'],
            json['namespace'],
            json['migrations'],
            json.get('shutdowns', []),
            json.get('inputs'),
            json['image'],
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

        create_passwords_script = f"""
set -ex

LC_ALL=C tr -dc '[:alnum:]' </dev/urandom | head -c 16 > {self.admin_password_file}
LC_ALL=C tr -dc '[:alnum:]' </dev/urandom | head -c 16 > {self.user_password_file}
"""

        create_database_script = f"""
set -ex

create_database_config={shq(json.dumps(create_database_config, indent=2))}
python3 create_database.py <<EOF
$create_database_config
EOF
"""

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
                self.image,
                command=['bash', '-c', create_passwords_script],
                attributes={'name': self.name + "_create_passwords"},
                output_files=[(x[1], x[0]) for x in password_files_input],
                parents=self.deps_parents(),
                regions=[REGION],
            )

        n_cores = 4 if scope == 'deploy' and not is_test_deployment else 1

        self.create_database_job = batch.create_job(
            self.image,
            command=['bash', '-c', create_database_script],
            attributes={'name': self.name},
            secrets=[
                {
                    'namespace': self.namespace,
                    'name': 'database-server-config',
                    'mount_path': '/sql-config',
                }
            ],
            service_account={'namespace': self.namespace, 'name': 'admin'},
            input_files=input_files,
            parents=[self.create_passwords_job] if self.create_passwords_job else self.deps_parents(),
            network='private',
            resources={'preemptible': False, 'cpu': str(n_cores)},
            regions=[REGION],
        )

    def cleanup(self, batch, scope, parents):
        pass
