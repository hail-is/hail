import abc
import os.path
import json
from shlex import quote as shq
import yaml
import jinja2
from .log import log
from .utils import flatten, generate_token
from .constants import BUCKET
from .environment import GCP_PROJECT, DOMAIN, IP, CI_UTILS_IMAGE


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
    def __init__(self, code, deploy, json, name_step):
        self.code = code
        self.deploy = deploy
        self.json = json
        self.name_step = name_step


class BuildConfiguration:
    def __init__(self, code, config_str, deploy):
        config = yaml.safe_load(config_str)
        name_step = {}
        self.steps = []
        for step_config in config['steps']:
            step_params = StepParameters(code, deploy, step_config, name_step)
            step = Step.from_json(step_params)
            self.steps.append(step)
            name_step[step.name] = step

    async def build(self, batch, code, deploy):
        if deploy:
            scope = 'deploy'
        else:
            scope = 'test'

        for step in self.steps:
            if step.scopes is None or scope in step.scopes:
                await step.build(batch, code, deploy)

        ids = set()
        for step in self.steps:
            ids.update(step.self_ids())
        ids = list(ids)

        sink = await batch.create_job('ubuntu:18.04',
                                      command=['/bin/true'],
                                      attributes={'name': 'sink'},
                                      parent_ids=ids)

        for step in self.steps:
            if step.scopes is None or scope in step.scopes:
                await step.cleanup(batch, deploy, sink)


class Step(abc.ABC):
    def __init__(self, params):
        json = params.json

        self.name = json['name']
        if 'dependsOn' in json:
            self.deps = [params.name_step[d] for d in json['dependsOn']]
        else:
            self.deps = None
        self.scopes = json.get('scopes')

        self.token = generate_token()

    def input_config(self, code, deploy):
        config = {}
        config['global'] = {
            'project': GCP_PROJECT,
            'domain': DOMAIN,
            'ip': IP
        }
        config['token'] = self.token
        config['deploy'] = deploy
        config['code'] = code.config()
        if self.deps:
            for d in self.deps:
                config[d.name] = d.config(deploy)
        return config

    def deps_parent_ids(self):
        if not self.deps:
            return None
        return flatten([d.self_ids() for d in self.deps])

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
        if kind == 'createDatabase':
            return CreateDatabaseStep.from_json(params)
        raise ValueError(f'unknown build step kind: {kind}')

    @abc.abstractmethod
    async def build(self, batch, code, deploy):
        pass


class BuildImageStep(Step):
    def __init__(self, params, dockerfile, context_path, publish_as, inputs):  # pylint: disable=unused-argument
        super().__init__(params)
        self.dockerfile = dockerfile
        self.context_path = context_path
        self.publish_as = publish_as
        self.inputs = inputs
        if params.deploy and publish_as:
            self.base_image = f'gcr.io/{GCP_PROJECT}/{self.publish_as}'
        else:
            self.base_image = f'gcr.io/{GCP_PROJECT}/ci-intermediate'
        self.image = f'{self.base_image}:{self.token}'
        self.job = None

    def self_ids(self):
        if self.job:
            return [self.job.id]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return BuildImageStep(params,
                              json['dockerFile'],
                              json.get('contextPath'),
                              json.get('publishAs'),
                              json.get('inputs'))

    def config(self, deploy):  # pylint: disable=unused-argument
        return {
            'token': self.token,
            'image': self.image
        }

    async def build(self, batch, code, deploy):
        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{BUCKET}/build/{batch.attributes["token"]}{i["from"]}', f'/io/{os.path.basename(i["to"])}'))
        else:
            input_files = None

        config = self.input_config(code, deploy)

        if self.context_path:
            context = f'repo/{self.context_path}'
            init_context = ''
        else:
            context = 'context'
            init_context = 'mkdir context'

        dockerfile = 'Dockerfile'
        render_dockerfile = f'python3 jinja2_render.py {shq(json.dumps(config))} {shq(f"repo/{self.dockerfile}")} Dockerfile'

        if self.publish_as:
            published_latest = shq(f'gcr.io/{GCP_PROJECT}/{self.publish_as}:latest')
            pull_published_latest = f'docker pull {shq(published_latest)} || true'
            cache_from_published_latest = f'--cache-from {shq(published_latest)}'
        else:
            pull_published_latest = ''
            cache_from_published_latest = ''

        push_image = f'''
time docker push {self.image}
'''
        if deploy and self.publish_as:
            push_image = f'''
docker tag {shq(self.image)} {self.base_image}:latest
docker push {self.base_image}:latest
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

rm -rf repo
mkdir repo
(cd repo; {code.checkout_script()})
{render_dockerfile}
{init_context}
{copy_inputs}

FROM_IMAGE=$(awk '$1 == "FROM" {{ print $2; exit }}' {shq(dockerfile)})

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key/gcr-push-service-account-key.json
gcloud -q auth configure-docker

docker pull $FROM_IMAGE
{pull_published_latest}
docker build --memory="1.5g" --cpus="1.0" -t {shq(self.image)} \
  -f {dockerfile} \
  --cache-from $FROM_IMAGE {cache_from_published_latest} \
  {context}
{push_image}

date
'''

        log.info(f'step {self.name}, script:\n{script}')

        volumes = [{
            'volume': {
                'name': 'docker-sock-volume',
                'hostPath': {
                    'path': '/var/run/docker.sock',
                    'type': 'File'
                }
            },
            'volume_mount': {
                'mountPath': '/var/run/docker.sock',
                'name': 'docker-sock-volume'
            }
        }, {
            'volume': {
                'name': 'gcr-push-service-account-key',
                'secret': {
                    'optional': False,
                    'secretName': 'gcr-push-service-account-key'
                }
            },
            'volume_mount': {
                'mountPath': '/secrets/gcr-push-service-account-key',
                'name': 'gcr-push-service-account-key',
                'readOnly': True
            }
        }]

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          resources={
                                              'requests': {
                                                  'memory': '2G',
                                                  'cpu': '1'
                                              },
                                              'limits': {
                                                  'memory': '2G',
                                                  'cpu': '1'
                                              }
                                          },
                                          attributes={'name': self.name},
                                          volumes=volumes,
                                          input_files=input_files,
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, deploy, sink):
        if deploy and self.publish_as:
            return

        volumes = [{
            'volume': {
                'name': 'gcr-push-service-account-key',
                'secret': {
                    'optional': False,
                    'secretName': 'gcr-push-service-account-key'
                }
            },
            'volume_mount': {
                'mountPath': '/secrets/gcr-push-service-account-key',
                'name': 'gcr-push-service-account-key',
                'readOnly': True
            }
        }]

        script = f'''
set -x
date

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key/gcr-push-service-account-key.json

gcloud -q container images untag {shq(self.image)}

date
true
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': f'cleanup_{self.name}'},
                                          volumes=volumes,
                                          parent_ids=[sink.id],
                                          always_run=True)


class RunImageStep(Step):
    def __init__(self, params, image, script, inputs, outputs, resources, service_account, secrets, always_run):  # pylint: disable=unused-argument
        super().__init__(params)
        self.image = expand_value_from(image, self.input_config(params.code, params.deploy))
        self.script = script
        self.inputs = inputs
        self.outputs = outputs
        self.resources = resources
        self.service_account = service_account
        self.secrets = secrets
        self.always_run = always_run
        self.job = None

    def self_ids(self):
        if self.job:
            return [self.job.id]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return RunImageStep(params,
                            json['image'],
                            json['script'],
                            json.get('inputs'),
                            json.get('outputs'),
                            json.get('resources'),
                            json.get('serviceAccount'),
                            json.get('secrets'),
                            json.get('alwaysRun', False))

    def config(self, deploy):  # pylint: disable=unused-argument
        return {
            'token': self.token
        }

    async def build(self, batch, code, deploy):
        template = jinja2.Template(self.script, undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        rendered_script = template.render(**self.input_config(code, deploy))

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

        volumes = []
        if self.secrets:
            for secret in self.secrets:
                name = secret['name']
                mount_path = secret['mountPath']
                volumes.append({
                    'volume': {
                        'name': name,
                        'secret': {
                            'optional': False,
                            'secretName': name
                        }
                    },
                    'volume_mount': {
                        'mountPath': mount_path,
                        'name': name
                    }
                })

        self.job = await batch.create_job(
            self.image,
            command=['bash', '-c', rendered_script],
            resources=self.resources,
            attributes={'name': self.name},
            input_files=input_files,
            output_files=output_files,
            volumes=volumes,
            service_account_name=self.service_account,
            parent_ids=self.deps_parent_ids(),
            always_run=self.always_run)

    async def cleanup(self, batch, deploy, sink):
        pass


class CreateNamespaceStep(Step):
    def __init__(self, params, namespace_name, admin_service_account, public, secrets):
        super().__init__(params)
        self.namespace_name = namespace_name
        if admin_service_account:
            self.admin_service_account = {
                'name': admin_service_account['name'],
                'namespace': get_namespace(admin_service_account['namespace'], self.input_config(params.code, params.deploy))
            }
        else:
            self.admin_service_account = None
        self.public = public
        self.secrets = secrets
        self.job = None
        if params.deploy:
            self._name = namespace_name
        else:
            self._name = f'{params.code.short_str()}-{namespace_name}-{self.token}'

    def self_ids(self):
        if self.job:
            return [self.job.id]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return CreateNamespaceStep(params,
                                   json['namespaceName'],
                                   json.get('adminServiceAccount'),
                                   json.get('public', False),
                                   json.get('secrets'))

    def config(self, deploy):
        conf = {
            'token': self.token,
            'kind': 'createNamespace',
            'name': self._name
        }
        if self.public:
            if deploy:
                conf['domain'] = DOMAIN
            else:
                conf['domain'] = 'internal'
        return conf

    async def build(self, batch, code, deploy):  # pylint: disable=unused-argument
        # FIXME label
        config = f'''\
apiVersion: v1
kind: Namespace
metadata:
  name: {self._name}
  labels:
    for: test
'''

        if self.admin_service_account:
            admin_service_account_name = self.admin_service_account['name']
            admin_service_account_namespace = self.admin_service_account['namespace']
            config = config + f'''\
---
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

        if self.secrets and not deploy:
            for s in self.secrets:
                script += f'''
kubectl -n {self.namespace_name} get -o json --export secret {s} | jq '.metadata.name = "{s}"' | kubectl -n {self._name} apply -f -
'''

        script += '''
date
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': self.name},
                                          # FIXME configuration
                                          service_account_name='ci-agent',
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, deploy, sink):
        if deploy:
            return

        script = f'''
set -x
date

kubectl delete namespace {self._name}

date
true
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': f'cleanup_{self.name}'},
                                          service_account_name='ci-agent',
                                          parent_ids=[sink.id],
                                          always_run=True)


class DeployStep(Step):
    def __init__(self, params, namespace, config_file, link, wait):  # pylint: disable=unused-argument
        super().__init__(params)
        self.namespace = get_namespace(namespace, self.input_config(params.code, params.deploy))
        self.config_file = config_file
        self.link = link
        self.wait = wait
        self.job = None

    def self_ids(self):
        if self.job:
            return [self.job.id]
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

    def config(self, deploy):  # pylint: disable=unused-argument
        return {
            'token': self.token
        }

    async def build(self, batch, code, deploy):
        with open(f'{code.repo_dir()}/{self.config_file}', 'r') as f:
            template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
            rendered_config = template.render(**self.input_config(code, deploy))

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
kubectl -n {self.namespace} logs -l app={name}
set -e
(exit $EC)
'''
                elif w['kind'] == 'Service':
                    assert w['for'] == 'alive', w['for']
                    port = w.get('port', 80)
                    timeout = w.get('timeout', 60)
                    script += f'''
set +e
kubectl -n {self.namespace} rollout status --timeout=1h deployment {name} && \
  kubectl -n {self.namespace} wait --timeout=1h --for=condition=available deployment {name} && \
  python3 wait-for.py {timeout} {self.namespace} Service -p {port} {name}
EC=$?
kubectl -n {self.namespace} logs -l app={name}
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
kubectl -n {self.namespace} logs {name}
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

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes=attrs,
                                          # FIXME configuration
                                          service_account_name='ci-agent',
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, deploy, sink):  # pylint: disable=unused-argument
        if self.wait:
            script = ''
            for w in self.wait:
                name = w['name']
                if w['kind'] == 'Deployment':
                    script += f'kubectl -n {self.namespace} logs -l app={name}\n'
                elif w['kind'] == 'Service':
                    assert w['for'] == 'alive', w['for']
                    script += f'kubectl -n {self.namespace} logs -l app={name}\n'
                else:
                    assert w['kind'] == 'Pod', w['kind']
                    script += f'kubectl -n {self.namespace} logs {name}\n'
            script += 'date\n'
            self.job = await batch.create_job(CI_UTILS_IMAGE,
                                              command=['bash', '-c', script],
                                              attributes={'name': self.name + '_logs'},
                                              # FIXME configuration
                                              service_account_name='ci-agent',
                                              parent_ids=[sink.id],
                                              always_run=True)


class CreateDatabaseStep(Step):
    def __init__(self, params, database_name, namespace):
        super().__init__(params)
        # FIXME validate
        self.database_name = database_name
        self.namespace = get_namespace(namespace, self.input_config(params.code, params.deploy))
        self.job = None

        # MySQL user name can be up to 16 characters long before MySQL 5.7.8 (32 after)
        if params.deploy:
            self._name = database_name
            self.admin_username = f'{self._name}-admin'
            self.user_username = f'{self._name}-user'
        else:
            self._name = f'{params.code.short_str()}-{database_name}-{self.token}'
            self.admin_username = generate_token()
            self.user_username = generate_token()

        self.admin_secret_name = f'sql-{self._name}-{self.admin_username}-config'
        self.user_secret_name = f'sql-{self._name}-{self.user_username}-config'

    def self_ids(self):
        if self.job:
            return [self.job.id]
        return []

    @staticmethod
    def from_json(params):
        json = params.json
        return CreateDatabaseStep(params,
                                  json['databaseName'],
                                  json['namespace'])

    def config(self, deploy):  # pylint: disable=unused-argument
        return {
            'token': self.token,
            'name': self._name,
            'admin_username': self.admin_username,
            'admin_secret_name': self.admin_secret_name,
            'user_username': self.user_username,
            'user_secret_name': self.user_secret_name
        }

    async def build(self, batch, code, deploy):  # pylint: disable=unused-argument
        if deploy:
            return

        script = f'''
set -e
echo date
date

ADMIN_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')
USER_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')

cat | mysql --host=10.80.0.3 -u root <<EOF
CREATE DATABASE \\`{self._name}\\`;

CREATE USER '{self.admin_username}'@'%' IDENTIFIED BY '$ADMIN_PASSWORD';
GRANT ALL ON \\`{self._name}\\`.* TO '{self.admin_username}'@'%';

CREATE USER '{self.user_username}'@'%' IDENTIFIED BY '$USER_PASSWORD';
GRANT SELECT, INSERT, UPDATE, DELETE ON \\`{self._name}\\`.* TO '{self.user_username}'@'%';
EOF

echo create database, admin and user...
echo "$SQL_SCRIPT" | mysql --host=10.80.0.3 -u root

echo create admin secret...
cat > sql-config.json <<EOF
{{
  "host": "10.80.0.3",
  "port": 3306,
  "user": "{self.admin_username}",
  "password": "$ADMIN_PASSWORD",
  "instance": "db-gh0um",
  "connection_name": "hail-vdc:us-central1:db-gh0um",
  "db": "{self._name}"
}}
EOF
cat > sql-config.cnf <<EOF
[client]
host=10.80.0.3
user={self.admin_username}
password="$ADMIN_PASSWORD"
database={self._name}
EOF
kubectl -n {shq(self.namespace)} create secret generic {shq(self.admin_secret_name)} --from-file=sql-config.json --from-file=sql-config.cnf

echo create user secret...
cat > sql-config.json <<EOF
{{
  "host": "10.80.0.3",
  "port": 3306,
  "user": "{self.user_username}",
  "password": "$USER_PASSWORD",
  "instance": "db-gh0um",
  "connection_name": "hail-vdc:us-central1:db-gh0um",
  "db": "{self._name}"
}}
EOF
cat > sql-config.cnf <<EOF
[client]
host=10.80.0.3
user={self.user_username}
password="$USER_PASSWORD"
database={self._name}
EOF
kubectl -n {shq(self.namespace)} create secret generic {shq(self.user_secret_name)} --from-file=sql-config.json --from-file=sql-config.cnf

echo database = {shq(self._name)}
echo admin_username = {shq(self.admin_username)}
echo admin_secret_name = {shq(self.admin_secret_name)}
echo user_username = {shq(self.user_username)}
echo user_secret_name = {shq(self.user_secret_name)}

echo date
date
echo done.
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': self.name},
                                          # FIXME configuration
                                          service_account_name='ci-agent',
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, deploy, sink):
        if deploy:
            return

        script = f'''
set -x
date

cat | mysql --host=10.80.0.3 -u root <<EOF
DROP DATABASE \\`{self._name}\\`;
DROP USER '{self.admin_username}';
DROP USER '{self.user_username}';
EOF

date
true
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': f'cleanup_{self.name}'},
                                          # FIXME configuration
                                          service_account_name='ci-agent',
                                          parent_ids=[sink.id],
                                          always_run=True)
