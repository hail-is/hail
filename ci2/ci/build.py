import os.path
import json
import string
import secrets
from shlex import quote as shq
import yaml
import jinja2
from .log import log
from .utils import flatten
from .constants import BUCKET
from .environment import GCP_PROJECT, DOMAIN, IP, CI_UTILS_IMAGE


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


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


class BuildConfiguration:
    def __init__(self, pr, config_str):
        config = yaml.safe_load(config_str)
        name_step = {}
        self.steps = []
        for step_config in config['steps']:
            step = Step.from_json(pr, step_config, name_step)
            self.steps.append(step)
            name_step[step.name] = step

    async def build(self, batch, pr):
        for step in self.steps:
            await step.build(batch, pr)

        ids = set()
        for step in self.steps:
            ids.update(step.self_ids())
        ids = list(ids)

        sink = await batch.create_job('ubuntu:18.04',
                                      command=['/bin/true'],
                                      attributes={'name': 'sink'},
                                      parent_ids=ids)

        for step in self.steps:
            await step.cleanup(batch, sink)


class Step:
    def __init__(self, name, deps):
        self.name = name
        self.deps = deps
        self.token = generate_token()

    def input_config(self, pr):
        config = {}
        config['global'] = {
            'project': GCP_PROJECT,
            'ip': IP
        }
        config['pr'] = pr.config()
        if self.deps:
            for d in self.deps:
                config[d.name] = d.config()
        return config

    def deps_parent_ids(self):
        if not self.deps:
            return None
        return flatten([d.self_ids() for d in self.deps])

    @staticmethod
    def from_json(pr, json, name_step):
        kind = json['kind']
        name = json['name']
        if 'dependsOn' in json:
            deps = [name_step[d] for d in json['dependsOn']]
        else:
            deps = None
        if kind == 'buildImage':
            return BuildImageStep.from_json(name, deps, json)
        if kind == 'runImage':
            return RunImageStep.from_json(pr, name, deps, json)
        if kind == 'createNamespace':
            return CreateNamespaceStep.from_json(pr, name, deps, json)
        if kind == 'deploy':
            return DeployStep.from_json(pr, name, deps, json)
        if kind == 'createDatabase':
            return CreateDatabaseStep.from_json(pr, name, deps, json)
        raise ValueError(f'unknown build step kind: {kind}')


class BuildImageStep(Step):
    def __init__(self, name, deps, dockerfile, context_path, publish_as, inputs):
        super().__init__(name, deps)
        self.dockerfile = dockerfile
        self.context_path = context_path
        self.publish_as = publish_as
        self.inputs = inputs
        self.image = f'gcr.io/{GCP_PROJECT}/ci-intermediate:{self.token}'
        self.job = None

    def self_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(name, deps, json):
        return BuildImageStep(name, deps,
                              json['dockerFile'],
                              json.get('contextPath'),
                              json.get('publishAs'),
                              json.get('inputs'))

    def config(self):
        return {'image': self.image}

    async def build(self, batch, pr):
        if self.inputs:
            input_files = []
            for i in self.inputs:
                input_files.append((f'{BUCKET}/build/{batch.attributes["token"]}{i["from"]}', f'/io/{os.path.basename(i["to"])}'))
        else:
            input_files = None

        config = self.input_config(pr)

        if self.context_path:
            context = f'repo/{self.context_path}'
            init_context = ''
        else:
            context = 'context'
            init_context = 'mkdir context'

        if self.dockerfile.endswith('.in'):
            dockerfile = 'Dockerfile'
            render_dockerfile = f'python3 jinja2_render.py {shq(json.dumps(config))} {shq(f"repo/{self.dockerfile}")} Dockerfile'
        else:
            dockerfile = f'repo/{self.dockerfile}'
            render_dockerfile = ''

        if self.publish_as:
            published_latest = shq(f'gcr.io/{GCP_PROJECT}/{self.publish_as}:latest')
            pull_published_latest = f'docker pull {shq(published_latest)} || true'
            cache_from_published_latest = f'--cache-from {shq(published_latest)}'
        else:
            pull_published_latest = ''
            cache_from_published_latest = ''

        copy_inputs = ''
        if self.inputs:
            for i in self.inputs:
                # to is relative to docker context
                copy_inputs = copy_inputs + f'''
mkdir -p {shq(os.path.dirname(f'{context}{i["to"]}'))}
mv -a {shq(f'/io/{os.path.basename(i["to"])}')} {shq(f'{context}{i["to"]}')}
'''

        script = f'''
set -ex

git clone {shq(pr.target_branch.branch.repo.url)} repo

git -C repo config user.email hail-ci-leader@example.com
git -C repo config user.name hail-ci-leader

git -C repo remote add {shq(pr.source_repo.short_str())} {shq(pr.source_repo.url)}
git -C repo fetch -q {shq(pr.source_repo.short_str())}
git -C repo checkout {shq(pr.target_branch.sha)}
git -C repo merge {shq(pr.source_sha)} -m 'merge PR'

{render_dockerfile}
{init_context}
{copy_inputs}

FROM_IMAGE=$(awk '$1 == "FROM" {{ print $2; exit }}' {shq(dockerfile)})

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key/gcr-push-service-account-key.json
gcloud -q auth configure-docker

docker pull $FROM_IMAGE
{pull_published_latest}
docker build -t {shq(self.image)} \
  -f {dockerfile} \
  --cache-from $FROM_IMAGE {cache_from_published_latest} \
  {context}
docker push {shq(self.image)}
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

        sa = None
        if self.inputs is not None:
            sa = 'ci2'

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': self.name},
                                          volumes=volumes,
                                          input_files=input_files,
                                          copy_service_account_name=sa,
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, sink):
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
gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key/gcr-push-service-account-key.json

gcloud -q container images delete {shq(self.image)}
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': f'cleanup_{self.name}'},
                                          volumes=volumes,
                                          parent_ids=[sink.id],
                                          always_run=True)


class RunImageStep(Step):
    def __init__(self, pr, name, deps, image, script, inputs, outputs):
        super().__init__(name, deps)
        self.image = expand_value_from(image, self.input_config(pr))
        self.script = script
        self.inputs = inputs
        self.outputs = outputs
        self.job = None

    def self_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(pr, name, deps, json):
        return RunImageStep(pr, name, deps,
                            json['image'],
                            json['script'],
                            json.get('inputs'),
                            json.get('outputs'))

    def config(self):  # pylint: disable=no-self-use
        return {}

    async def build(self, batch, pr):
        template = jinja2.Template(self.script, undefined=jinja2.StrictUndefined)
        rendered_script = template.render(**self.input_config(pr))

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

        sa = None
        if (self.outputs is not None) or (self.inputs is not None):
            sa = 'ci2'

        self.job = await batch.create_job(
            self.image,
            command=['bash', '-c', rendered_script],
            attributes={'name': self.name},
            input_files=input_files,
            output_files=output_files,
            copy_service_account_name=sa,
            parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, sink):
        pass


class CreateNamespaceStep(Step):
    def __init__(self, pr, name, deps, namespace_name, admin_service_account, public, secrets):
        super().__init__(name, deps)
        self.namespace_name = namespace_name
        if admin_service_account:
            self.admin_service_account = {
                'name': admin_service_account['name'],
                'namespace': get_namespace(admin_service_account['namespace'], self.input_config(pr))
            }
        else:
            self.admin_service_account = None
        self.public = public
        self.secrets = secrets
        self.job = None
        self._name = f'test-{pr.number}-{namespace_name}-{self.token}'

    def self_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(pr, name, deps, json):
        return CreateNamespaceStep(pr, name, deps,
                                   json['namespaceName'],
                                   json.get('adminServiceAccount'),
                                   json.get('public', False),
                                   json.get('secrets'))

    def config(self):
        conf = {
            'kind': 'createNamespace',
            'name': self._name
        }
        if self.public:
            conf['domain'] = f'{self._name}.internal.{DOMAIN}'
        return conf

    async def build(self, batch, pr):  # pylint: disable=unused-argument
        # FIXME label
        config = f'''\
apiVersion: v1
kind: Namespace
metadata:
  name: {self._name}
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

cat | kubectl apply -f - <<EOF
{config}
EOF
'''

        if self.secrets:
            for s in self.secrets:
                script = script + f'''
kubectl -n {self.namespace_name} get -o json --export secret {s} | kubectl -n {self._name} apply -f -
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': self.name},
                                          # FIXME configuration
                                          service_account_name='ci2-agent',
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, sink):
        script = f'''
kubectl delete namespace {self._name}
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': f'cleanup_{self.name}'},
                                          service_account_name='ci2-agent',
                                          parent_ids=[sink.id],
                                          always_run=True)


class DeployStep(Step):
    def __init__(self, pr, name, deps, namespace, config_file, link, wait):
        super().__init__(name, deps)
        # FIXME check available namespaces
        self.namespace = get_namespace(namespace, self.input_config(pr))
        self.config_file = config_file
        self.link = link
        self.wait = wait
        self.job = None

    def self_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(pr, name, deps, json):
        return DeployStep(pr, name, deps,
                          json['namespace'],
                          # FIXME config_file
                          json['config'],
                          json.get('link'),
                          json.get('wait'))

    def config(self):  # pylint: disable=no-self-use
        return {}

    async def build(self, batch, pr):
        target_repo = pr.target_branch.branch.repo
        repo_dir = f'repos/{target_repo.short_str()}'

        with open(f'{repo_dir}/{self.config_file}', 'r') as f:
            rendered_config = f.read()
            if self.config_file.endswith('.in'):
                template = jinja2.Template(rendered_config, undefined=jinja2.StrictUndefined)
                rendered_config = template.render(**self.input_config(pr))

        script = f'''
set -ex

cat | kubectl apply -n {self.namespace} -f - <<EOF
{rendered_config}
EOF
'''

        if self.wait:
            for w in self.wait:
                name = w['name']
                if w['kind'] == 'Deployment':
                    assert w['for'] == 'available', w['for']
                    # FIXME what if the cluster isn't big enough?
                    script = script + f'''
kubectl -n {self.namespace} wait --timeout=600s deployment --for=condition=available {name}
'''
                else:
                    assert w['kind'] == 'Pod', w['kind']
                    if w['for'] == 'ready':
                        script = script + f'''
kubectl -n {self.namespace} wait --timeout=600s pod --for=condition=ready {name}
'''
                    else:
                        assert w['for'] == 'completed', w['for']
                        script = script + f'''
set +e
python3 wait-for-pod.py 600 {self.namespace} {name}
EC=$?
kubectl -n {self.namespace} logs {name}
set -e
(exit $EC)
'''

        attrs = {'name': self.name}
        if self.link is not None:
            attrs['link'] = ','.join(self.link)
            attrs['domain'] = f'{self.namespace}.internal.{DOMAIN}'

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes=attrs,
                                          # FIXME configuration
                                          service_account_name='ci2-agent',
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, sink):
        # namespace cleanup will handle deployments
        pass


class CreateDatabaseStep(Step):
    def __init__(self, pr, name, deps, database_name, namespace):
        super().__init__(name, deps)
        # FIXME validate
        self.database_name = database_name
        self.namespace = get_namespace(namespace, self.input_config(pr))
        self.job = None
        self._name = f'test-{pr.number}-{database_name}-{self.token}'
        # MySQL user name can be up to 16 characters long before MySQL 5.7.8 (32 after)
        self.admin_username = generate_token()
        self.admin_secret_name = f'sql-{self._name}-{self.admin_username}-config'
        self.user_username = generate_token()
        self.user_secret_name = f'sql-{self._name}-{self.user_username}-config'

    def self_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(pr, name, deps, json):
        return CreateDatabaseStep(pr, name, deps,
                                  json['databaseName'],
                                  json['namespace'])

    def config(self):
        return {
            'name': self._name,
            'admin_username': self.admin_username,
            'admin_secret_name': self.admin_secret_name,
            'user_username': self.user_username,
            'user_secret_name': self.user_secret_name
        }

    async def build(self, batch, pr):  # pylint: disable=unused-argument
        script = f'''
set -e

ADMIN_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')
USER_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')

cat | mysql --host=10.80.0.3 -u root <<EOF
CREATE DATABASE `{self._name}`;

CREATE USER '{self.admin_username}'@'%' IDENTIFIED BY '$ADMIN_PASSWORD';
GRANT ALL ON `{self._name}`.* TO '{self.admin_username}'@'%';

CREATE USER '{self.user_username}'@'%' IDENTIFIED BY '$USER_PASSWORD';
GRANT SELECT, INSERT, UPDATE, DELETE ON `{self._name}`.* TO '{self.user_username}'@'%';
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
kubectl -n {shq(self.namespace)} create secret generic {shq(self.admin_secret_name)} --from-file=sql-config.json

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
kubectl -n {shq(self.namespace)} create secret generic {shq(self.user_secret_name)} --from-file=sql-config.json

echo database = {shq(self._name)}
echo admin_username = {shq(self.admin_username)}
echo admin_secret_name = {shq(self.admin_secret_name)}
echo user_username = {shq(self.user_username)}
echo user_secret_name = {shq(self.user_secret_name)}

echo done.
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': self.name},
                                          # FIXME configuration
                                          service_account_name='ci2-agent',
                                          parent_ids=self.deps_parent_ids())

    async def cleanup(self, batch, sink):
        script = f'''
cat | mysql --host=10.80.0.3 -u root <<EOF
DROP DATABASE `{self._name}`;
DROP USER '{self.admin_username}';
DROP USER '{self.user_username}';
EOF
'''

        self.job = await batch.create_job(CI_UTILS_IMAGE,
                                          command=['bash', '-c', script],
                                          attributes={'name': f'cleanup_{self.name}'},
                                          # FIXME configuration
                                          service_account_name='ci2-agent',
                                          parent_ids=[sink.id],
                                          always_run=True)
