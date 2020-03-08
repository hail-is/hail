import abc
import os.path
import json
import logging
from collections import defaultdict
from shlex import quote as shq
import yaml
import jinja2
from hailtop.utils import RETRY_FUNCTION_SCRIPT
from .utils import flatten, generate_token
from .constants import BUCKET
from .environment import GCP_PROJECT, GCP_ZONE, DOMAIN, IP, CI_UTILS_IMAGE, \
    DEFAULT_NAMESPACE, BATCH_PODS_NAMESPACE, KUBERNETES_SERVER_URL
from .globals import is_test_deployment

log = logging.getLogger('ci')


pretty_print_log = "jq -Rr '. as $raw | try \
(fromjson | if .hail_log == 1 then \
    ([.levelname, .asctime, .filename, .funcNameAndLine, .message, .exc_info] | @tsv) \
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


class BuildConfiguration:
    def __init__(self, code, config_str, scope, requested_step_names=()):
        config = yaml.safe_load(config_str)
        name_step = {}
        self.steps = []

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

            def request(step):
                if step not in visited:
                    visited.add(step)
                    for s2 in step.deps:
                        request(s2)

            for step_name in requested_step_names:
                request(name_step[step_name])
            self.steps = [s for s in self.steps if s in visited]

    def build(self, batch, code, scope):
        assert scope in ('deploy', 'test', 'dev')

        for step in self.steps:
            if step.scopes is None or scope in step.scopes:
                step.build(batch, code, scope)

        if scope == 'dev':
            return

        step_to_parent_steps = defaultdict(set)
        for step in self.steps:
            for dep in step.all_deps():
                step_to_parent_steps[dep].add(step)

        for step in self.steps:
            parent_jobs = flatten([parent_step.wrapped_job() for parent_step in step_to_parent_steps[step]])

            log.info(f"Cleanup {step.name} after running {[parent_step.name for parent_step in step_to_parent_steps[step]]}")

            if step.scopes is None or scope in step.scopes:
                step.cleanup(batch, scope, parent_jobs)


class Step(abc.ABC):
    def __init__(self, params):
        json = params.json

        self.name = json['name']
        if 'dependsOn' in json:
            self.deps = [params.name_step[d] for d in json['dependsOn'] if params.name_step[d]]
        else:
            self.deps = []
        self.scopes = json.get('scopes')
        self.run_if_requested = json.get('runIfRequested', False)

        self.token = generate_token()

    def input_config(self, code, scope):
        config = {}
        config['global'] = {
            'project': GCP_PROJECT,
            'zone': GCP_ZONE,
            'domain': DOMAIN,
            'ip': IP,
            'k8s_server_url': KUBERNETES_SERVER_URL
        }
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
            return BuildImageStep.from_json(params)
        if kind == 'runImage':
            return RunImageStep.from_json(params)
        if kind == 'createNamespace':
            return CreateNamespaceStep.from_json(params)
        if kind == 'deploy':
            return DeployStep.from_json(params)
        if kind in ('createDatabase', 'createDatabase2'):
            return CreateDatabaseStep.from_json(params)
        raise ValueError(f'unknown build step kind: {kind}')

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


class BuildImageStep(Step):
    def __init__(self, params, dockerfile, context_path, publish_as, inputs):  # pylint: disable=unused-argument
        super().__init__(params)
        self.dockerfile = dockerfile
        self.context_path = context_path
        self.publish_as = publish_as
        self.inputs = inputs
        if params.scope == 'deploy' and publish_as and not is_test_deployment:
            self.base_image = f'gcr.io/{GCP_PROJECT}/{self.publish_as}'
        else:
            self.base_image = f'gcr.io/{GCP_PROJECT}/ci-intermediate'
        self.image = f'{self.base_image}:{self.token}'
        self.job = None

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return BuildImageStep(params,
                              json['dockerFile'],
                              json.get('contextPath'),
                              json.get('publishAs'),
                              json.get('inputs'))

    def config(self, scope):  # pylint: disable=unused-argument
        return {
            'token': self.token,
            'image': self.image
        }

    def build(self, batch, code, scope):
        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{BUCKET}/build/{batch.attributes["token"]}{i["from"]}', f'/io/{os.path.basename(i["to"])}'))
        else:
            input_files = None

        config = self.input_config(code, scope)

        if self.context_path:
            context = f'repo/{self.context_path}'
            init_context = ''
        else:
            context = 'context'
            init_context = 'mkdir context'

        rendered_dockerfile = 'Dockerfile'
        if isinstance(self.dockerfile, dict):
            assert ['inline'] == list(self.dockerfile.keys())
            render_dockerfile = f'echo {shq(self.dockerfile["inline"])} > Dockerfile.{self.token};\n'
            unrendered_dockerfile = f'Dockerfile.{self.token}'
        else:
            assert isinstance(self.dockerfile, str)
            render_dockerfile = ''
            unrendered_dockerfile = f'repo/{self.dockerfile}'
        render_dockerfile += (f'python3 jinja2_render.py {shq(json.dumps(config))} '
                              f'{shq(unrendered_dockerfile)} {shq(rendered_dockerfile)}')

        if self.publish_as:
            published_latest = shq(f'gcr.io/{GCP_PROJECT}/{self.publish_as}:latest')
            pull_published_latest = f'retry docker pull {shq(published_latest)} || true'
            cache_from_published_latest = f'--cache-from {shq(published_latest)}'
        else:
            pull_published_latest = ''
            cache_from_published_latest = ''

        push_image = f'''
time retry docker push {self.image}
'''
        if scope == 'deploy' and self.publish_as and not is_test_deployment:
            push_image = f'''
docker tag {shq(self.image)} {self.base_image}:latest
retry docker push {self.base_image}:latest
''' + push_image

        copy_inputs = ''
        if self.inputs:
            for i in self.inputs:
                # to is relative to docker context
                copy_inputs = copy_inputs + f'''
mkdir -p {shq(os.path.dirname(f'{context}{i["to"]}'))}
cp {shq(f'/io/{os.path.basename(i["to"])}')} {shq(f'{context}{i["to"]}')}
'''

        script = f'''
set -ex
date

{ RETRY_FUNCTION_SCRIPT }

rm -rf repo
mkdir repo
(cd repo; {code.checkout_script()})
{render_dockerfile}
{init_context}
{copy_inputs}

FROM_IMAGE=$(awk '$1 == "FROM" {{ print $2; exit }}' {shq(rendered_dockerfile)})

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key/gcr-push-service-account-key.json
gcloud -q auth configure-docker

retry docker pull $FROM_IMAGE
{pull_published_latest}
docker build --memory="1.5g" --cpu-period=100000 --cpu-quota=100000 -t {shq(self.image)} \
  -f {rendered_dockerfile} \
  --cache-from $FROM_IMAGE {cache_from_published_latest} \
  {context}
{push_image}

date
'''

        log.info(f'step {self.name}, script:\n{script}')

        self.job = batch.create_job(CI_UTILS_IMAGE,
                                    command=['bash', '-c', script],
                                    mount_docker_socket=True,
                                    secrets=[{
                                        'namespace': BATCH_PODS_NAMESPACE,
                                        'name': 'gcr-push-service-account-key',
                                        'mount_path': '/secrets/gcr-push-service-account-key'
                                    }],
                                    resources={
                                        'memory': '2G',
                                        'cpu': '1'
                                    },
                                    attributes={'name': self.name},
                                    input_files=input_files,
                                    parents=self.deps_parents())

    def cleanup(self, batch, scope, parents):
        if scope == 'deploy' and self.publish_as and not is_test_deployment:
            return

        script = f'''
set -x
date

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key/gcr-push-service-account-key.json

gcloud -q container images untag {shq(self.image)}

date
true
'''

        self.job = batch.create_job(CI_UTILS_IMAGE,
                                    command=['bash', '-c', script],
                                    attributes={'name': f'cleanup_{self.name}'},
                                    secrets=[{
                                        'namespace': BATCH_PODS_NAMESPACE,
                                        'name': 'gcr-push-service-account-key',
                                        'mount_path': '/secrets/gcr-push-service-account-key'
                                    }],
                                    parents=parents,
                                    always_run=True)


class RunImageStep(Step):
    def __init__(self, params, image, script, inputs, outputs, port, resources, service_account, secrets, always_run, timeout):  # pylint: disable=unused-argument
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
                'namespace': get_namespace(service_account['namespace'], self.input_config(params.code, params.scope))
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
        return RunImageStep(params,
                            json['image'],
                            json['script'],
                            json.get('inputs'),
                            json.get('outputs'),
                            json.get('port'),
                            json.get('resources'),
                            json.get('serviceAccount'),
                            json.get('secrets'),
                            json.get('alwaysRun', False),
                            json.get('timeout', 3600))

    def config(self, scope):  # pylint: disable=unused-argument
        return {
            'token': self.token
        }

    def build(self, batch, code, scope):
        template = jinja2.Template(self.script, undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        rendered_script = template.render(**self.input_config(code, scope))

        log.info(f'step {self.name}, rendered script:\n{rendered_script}')

        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{BUCKET}/build/{batch.attributes["token"]}{i["from"]}', i["to"]))
        else:
            input_files = None

        if self.outputs:
            output_files = []
            for o in self.outputs:
                output_files.append((o["from"], f'{BUCKET}/build/{batch.attributes["token"]}{o["to"]}'))
        else:
            output_files = None

        secrets = []
        if self.secrets:
            for secret in self.secrets:
                namespace = get_namespace(secret['namespace'], self.input_config(code, scope))
                name = expand_value_from(secret['name'], self.input_config(code, scope))
                mount_path = secret['mountPath']
                secrets.append({
                    'namespace': namespace,
                    'name': name,
                    'mount_path': mount_path
                })

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
            timeout=self.timeout)

    def cleanup(self, batch, scope, parents):
        pass


class CreateNamespaceStep(Step):
    def __init__(self, params, namespace_name, admin_service_account, public, secrets):
        super().__init__(params)
        self.namespace_name = namespace_name
        if admin_service_account:
            self.admin_service_account = {
                'name': admin_service_account['name'],
                'namespace': get_namespace(admin_service_account['namespace'], self.input_config(params.code, params.scope))
            }
        else:
            self.admin_service_account = None
        self.public = public
        self.secrets = secrets
        self.job = None

        if is_test_deployment:
            self._name = DEFAULT_NAMESPACE
            return

        if params.scope == 'deploy':
            self._name = namespace_name
        elif params.scope == 'test':
            self._name = f'{params.code.short_str()}-{namespace_name}-{self.token}'
        elif params.scope == 'dev':
            self._name = params.code.namespace
        else:
            raise ValueError(f"{params.scope} is not a valid scope for creating namespace")

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return CreateNamespaceStep(params,
                                   json['namespaceName'],
                                   json.get('adminServiceAccount'),
                                   json.get('public', False),
                                   json.get('secrets'))

    def config(self, scope):  # pylint: disable=unused-argument
        return {
            'token': self.token,
            'kind': 'createNamespace',
            'name': self._name
        }

    def build(self, batch, code, scope):  # pylint: disable=unused-argument
        if is_test_deployment:
            return

        config = ""
        if scope in ['deploy', 'test']:
            # FIXME label
            config = config + f'''\
apiVersion: v1
kind: Namespace
metadata:
  name: {self._name}
  labels:
    for: test
---
'''
        config = config + f'''\
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

        if self.admin_service_account:
            admin_service_account_name = self.admin_service_account['name']
            admin_service_account_namespace = self.admin_service_account['namespace']
            config = config + f'''\
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

        if self.public:
            config = config + '''\
---
apiVersion: v1
kind: Service
metadata:
  name: router
  labels:
    app: router
spec:
  ports:
  - name: http
    port: 80
    protocol: TCP
    targetPort: 80
  selector:
    app: router
'''

        script = f'''
set -ex
date

echo {shq(config)} | kubectl apply -f -
'''

        if self.secrets and not scope == 'deploy':
            for s in self.secrets:
                script += f'''
kubectl -n {self.namespace_name} get -o json --export secret {s} | jq '.metadata.name = "{s}"' | kubectl -n {self._name} apply -f -
'''

        script += '''
date
'''

        self.job = batch.create_job(CI_UTILS_IMAGE,
                                    command=['bash', '-c', script],
                                    attributes={'name': self.name},
                                    # FIXME configuration
                                    service_account={
                                        'namespace': BATCH_PODS_NAMESPACE,
                                        'name': 'ci-agent'
                                    },
                                    parents=self.deps_parents())

    def cleanup(self, batch, scope, parents):
        if scope in ['deploy', 'dev'] or is_test_deployment:
            return

        script = f'''
set -x
date

kubectl delete namespace {self._name}

date
true
'''

        self.job = batch.create_job(CI_UTILS_IMAGE,
                                    command=['bash', '-c', script],
                                    attributes={'name': f'cleanup_{self.name}'},
                                    service_account={
                                        'namespace': BATCH_PODS_NAMESPACE,
                                        'name': 'ci-agent'
                                    },
                                    parents=parents,
                                    always_run=True)


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
        return DeployStep(params,
                          json['namespace'],
                          # FIXME config_file
                          json['config'],
                          json.get('link'),
                          json.get('wait'))

    def config(self, scope):  # pylint: disable=unused-argument
        return {
            'token': self.token
        }

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
kubectl -n {self.namespace} logs --tail=999999 -l app={name} | {pretty_print_log}
set -e
(exit $EC)
'''
                elif w['kind'] == 'Service':
                    assert w['for'] == 'alive', w['for']
                    resource_type = w.get('resource_type', 'deployment').lower()
                    timeout = w.get('timeout', 60)
                    if resource_type == 'statefulset':
                        wait_cmd = f'kubectl -n {self.namespace} wait --timeout=1h --for=condition=ready pods --selector=app={name}'
                    else:
                        assert resource_type == 'deployment'
                        wait_cmd = f'kubectl -n {self.namespace} wait --timeout=1h --for=condition=available deployment {name}'

                    script += f'''
set +e
kubectl -n {self.namespace} rollout status --timeout=1h {resource_type} {name} && \
  {wait_cmd}
EC=$?
kubectl -n {self.namespace} logs --tail=999999 -l app={name} | {pretty_print_log}
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
kubectl -n {self.namespace} logs --tail=999999 {name} | {pretty_print_log}
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

        self.job = batch.create_job(CI_UTILS_IMAGE,
                                    command=['bash', '-c', script],
                                    attributes=attrs,
                                    # FIXME configuration
                                    service_account={
                                        'namespace': BATCH_PODS_NAMESPACE,
                                        'name': 'ci-agent'
                                    },
                                    parents=self.deps_parents())

    def cleanup(self, batch, scope, parents):  # pylint: disable=unused-argument
        if self.wait:
            script = ''
            for w in self.wait:
                name = w['name']
                if w['kind'] == 'Deployment':
                    script += f'kubectl -n {self.namespace} logs --tail=999999 -l app={name} | {pretty_print_log}\n'
                elif w['kind'] == 'Service':
                    assert w['for'] == 'alive', w['for']
                    script += f'kubectl -n {self.namespace} logs --tail=999999 -l app={name} | {pretty_print_log}\n'
                else:
                    assert w['kind'] == 'Pod', w['kind']
                    script += f'kubectl -n {self.namespace} logs --tail=999999 {name} | {pretty_print_log}\n'
            script += 'date\n'
            self.job = batch.create_job(CI_UTILS_IMAGE,
                                        command=['bash', '-c', script],
                                        attributes={'name': self.name + '_logs'},
                                        # FIXME configuration
                                        service_account={
                                            'namespace': BATCH_PODS_NAMESPACE,
                                            'name': 'ci-agent'
                                        },
                                        parents=parents,
                                        always_run=True)


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
        self.job = None

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

        self.admin_secret_name = f'sql-{self.database_name}-admin-config'
        self.user_secret_name = f'sql-{self.database_name}-user-config'

    def wrapped_job(self):
        if self.job:
            return [self.job]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return CreateDatabaseStep(params,
                                  json['databaseName'],
                                  json['namespace'],
                                  json['migrations'],
                                  json.get('shutdowns', []),
                                  json.get('inputs'))

    def config(self, scope):  # pylint: disable=unused-argument
        return {
            'token': self.token,
            'admin_secret_name': self.admin_secret_name,
            'user_secret_name': self.user_secret_name
        }

    def build(self, batch, code, scope):  # pylint: disable=unused-argument
        create_database_config = {
            'namespace': self.namespace,
            'scope': scope,
            'database_name': self.database_name,
            '_name': self._name,
            'admin_username': self.admin_username,
            'user_username': self.user_username,
            'cant_create_database': self.cant_create_database,
            'migrations': self.migrations,
            'shutdowns': self.shutdowns
        }

        create_script = f'''
set -ex

python3 create_database.py {shq(json.dumps(create_database_config))}
'''

        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{BUCKET}/build/{batch.attributes["token"]}{i["from"]}', i["to"]))
        else:
            input_files = None

        self.job = batch.create_job(CI_UTILS_IMAGE,
                                    command=['bash', '-c', create_script],
                                    attributes={'name': self.name},
                                    secrets=[{
                                        'namespace': self.database_server_config_namespace,
                                        'name': 'database-server-config',
                                        'mount_path': '/sql-config'
                                    }],
                                    service_account={
                                        'namespace': BATCH_PODS_NAMESPACE,
                                        'name': 'ci-agent'
                                    },
                                    input_files=input_files,
                                    parents=self.deps_parents())

    def cleanup(self, batch, scope, parents):
        if scope in ['deploy', 'dev']:
            return

        cleanup_script = f'''
set -ex

cat | mysql --defaults-extra-file=/sql-config/sql-config.cnf <<EOF
DROP DATABASE \\`{self._name}\\`;
DROP USER '{self.admin_username}';
DROP USER '{self.user_username}';
EOF
'''

        self.job = batch.create_job(CI_UTILS_IMAGE,
                                    command=['bash', '-c', cleanup_script],
                                    attributes={'name': f'cleanup_{self.name}'},
                                    secrets=[{
                                        'namespace': self.database_server_config_namespace,
                                        'name': 'database-server-config',
                                        'mount_path': '/sql-config'
                                    }],
                                    service_account={
                                        'namespace': BATCH_PODS_NAMESPACE,
                                        'name': 'ci-agent'
                                    },
                                    parents=parents,
                                    always_run=True)
