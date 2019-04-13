import json
import string
import secrets
from shlex import quote as shq
import yaml
import jinja2
from .log import log
from .utils import flatten
from .environment import GCP_PROJECT, DOMAIN, IP


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


def expand_value_from(value, config):
    if not isinstance(value, dict):
        return value

    path = value['valueFrom']
    path = path.split('.')
    v = config
    for field in path:
        v = v[field]
    return v


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


class Step:
    def __init__(self, name, deps):
        self.name = name
        self.deps = deps
        self.token = generate_token()

    def input_config(self, pr):
        config = {}
        config['global'] = {
            'project': GCP_PROJECT,
            'domain': DOMAIN,
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
        return flatten([d.parent_ids() for d in self.deps])

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
    def __init__(self, name, deps, dockerfile, context_path, publish_as):
        super().__init__(name, deps)
        self.dockerfile = dockerfile
        self.context_path = context_path
        self.publish_as = publish_as
        self.image = f'gcr.io/{GCP_PROJECT}/ci-intermediate:{self.token}'
        self.job = None

    def parent_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(name, deps, json):
        return BuildImageStep(name, deps,
                              json['dockerFile'],
                              json.get('contextPath'),
                              json.get('publishAs'))

    def config(self):
        return {'image': self.image}

    async def build(self, batch, pr):
        config = self.input_config(pr)

        if self.context_path:
            context = f'repo/{self.context_path}'
            init_context = 'true'
        else:
            context = 'context'
            init_context = 'mkdir context'

        if self.dockerfile.endswith('.in'):
            dockerfile = 'Dockerfile'
            print('config', config)
            render_dockerfile = f'python3 jinja2_render.py {shq(json.dumps(config))} {shq(f"repo/{self.dockerfile}")} Dockerfile'
        else:
            dockerfile = f'repo/{self.dockerfile}'
            render_dockerfile = 'true'

        if self.publish_as:
            published_latest = shq(f'gcr.io/{GCP_PROJECT}/{self.publish_as}:latest')
            pull_published_latest = f'docker pull {shq(published_latest)} || true'
            cache_from_published_latest = f'--cache-from {shq(published_latest)}'
        else:
            pull_published_latest = 'true'
            cache_from_published_latest = ''

        script = f'''
set -ex

gcloud -q auth activate-service-account \
  --key-file=/secrets/gcr-push-service-account-key/gcr-push-service-account-key.json
gcloud -q auth configure-docker

git clone {shq(pr.target_branch.branch.repo.url)} repo

git -C repo config user.email hail-ci-leader@example.com
git -C repo config user.name hail-ci-leader

git -C repo remote add {shq(pr.source_repo.short_str())} {shq(pr.source_repo.url)}
git -C repo fetch -q {shq(pr.source_repo.short_str())}
git -C repo checkout {shq(pr.target_branch.sha)}
git -C repo merge {shq(pr.source_sha)} -m 'merge PR'

{render_dockerfile}
FROM_IMAGE=$(awk '$1 == "FROM" {{ print $2; exit }}' {shq(dockerfile)})
docker pull $FROM_IMAGE

{pull_published_latest}
{init_context}
docker build -t {shq(self.image)} \
  -f {dockerfile} \
  --cache-from $FROM_IMAGE {cache_from_published_latest} \
  {context}
docker push {shq(self.image)}
'''

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

        # FIXME image version
        self.job = await batch.create_job(f'gcr.io/{GCP_PROJECT}/ci-utils',
                                          command=['bash', '-c', script],
                                          volumes=volumes,
                                          parent_ids=self.deps_parent_ids())


class RunImageStep(Step):
    def __init__(self, pr, name, deps, image, script):
        super().__init__(name, deps)
        self.image = expand_value_from(image, self.input_config(pr))
        self.script = script
        self.job = None

    def parent_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(pr, name, deps, json):
        return RunImageStep(pr, name, deps,
                            json['image'],
                            json['script'])

    def config(self):  # pylint: disable=no-self-use
        return {}

    async def build(self, batch, pr):
        template = jinja2.Template(self.script, undefined=jinja2.StrictUndefined)
        rendered_script = template.render(**self.input_config(pr))

        log.info(f'step {self.name}, rendered script:\n{rendered_script}')

        self.job = await batch.create_job(
            self.image,
            command=['bash', '-c', rendered_script],
            parent_ids=self.deps_parent_ids())


class CreateNamespaceStep(Step):
    def __init__(self, pr, name, deps, namespace_name, admin_sa, public):
        super().__init__(name, deps)
        self.namespace_name = namespace_name
        if admin_sa:
            self.admin_sa = {
                'name': admin_sa['name'],
                # FIXME check
                'namespace': expand_value_from(admin_sa['namespace'], self.input_config(pr))
            }
        else:
            self.admin_sa = None
        self.public = public
        self.job = None
        self._name = f'test-{pr.number}-{namespace_name}-{self.token}'

    def parent_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(pr, name, deps, json):
        return CreateNamespaceStep(pr, name, deps,
                                   json['namespaceName'],
                                   json.get('adminServiceAccount'),
                                   json.get('public', False))

    def config(self):
        return {'name': self._name}

    async def build(self, batch, pr):  # pylint: disable=unused-argument
        # FIXME label
        config = f'''\
apiVersion: v1
kind: Namespace
metadata:
  name: {self._name}
  labels:
    for: test
'''

        if self.admin_sa:
            admin_sa_name = self.admin_sa['name']
            admin_sa_namespace = self.admin_sa['namespace']
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
kind: RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: {admin_sa_name}-{self.namespace_name}-admin-binding
  namespace: {self._name}
subjects:
- kind: ServiceAccount
  name: {admin_sa_name}
  namespace: {admin_sa_namespace}
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

        self.job = await batch.create_job(f'gcr.io/{GCP_PROJECT}/ci-utils',
                                          command=['bash', '-c', 'echo "$CONFIG" | kubectl apply -f -'],
                                          env={'CONFIG': config},
                                          # FIXME configuration
                                          service_account_name='ci2-agent',
                                          parent_ids=self.deps_parent_ids())


class DeployStep(Step):
    def __init__(self, pr, name, deps, namespace, config_file):
        super().__init__(name, deps)
        # FIXME check available namespaces
        self.namespace = expand_value_from(namespace, self.input_config(pr))
        self.config_file = config_file
        self.job = None

    def parent_ids(self):
        return [self.job.id]

    @staticmethod
    def from_json(pr, name, deps, json):
        return DeployStep(pr, name, deps,
                          json['namespace'],
                          # FIXME config_file
                          json['config'])

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

        self.job = await batch.create_job(f'gcr.io/{GCP_PROJECT}/ci-utils',
                                          command=['bash', '-c', f'echo "$CONFIG" | kubectl apply -n {self.namespace} -f -'],
                                          env={'CONFIG': rendered_config},
                                          # FIXME configuration
                                          service_account_name='ci2-agent',
                                          parent_ids=self.deps_parent_ids())


class CreateDatabaseStep(Step):
    def __init__(self, pr, name, deps, database_name, namespace):
        super().__init__(name, deps)
        # FIXME validate
        self.database_name = database_name
        # FIXME check namespace
        self.namespace = expand_value_from(namespace, self.input_config(pr))
        self.job = None
        self._name = f'test-{pr.number}-{database_name}-{self.token}'
        # MySQL user name can be up to 16 characters long before MySQL 5.7.8 (32 after)
        self.admin_username = generate_token()
        self.admin_password = secrets.token_urlsafe(16)
        self.admin_secret = f'sql-{self._name}-{self.admin_username}-config'
        self.user_username = generate_token()
        self.user_password = secrets.token_urlsafe(16)
        self.user_secret = f'sql-{self._name}-{self.user_username}-config'

    def parent_ids(self):
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
            'admin_secret': self.admin_secret,
            'user_username': self.user_username,
            'user_secret': self.user_secret
        }

    async def build(self, batch, pr):  # pylint: disable=unused-argument
        # FIXME are passwords in pod configuration?
        # FIXME create credentials programatically and clean up ourselves?
        sql_script = f'''
CREATE DATABASE `{self._name}`;

CREATE USER '{self.admin_username}'@'%' IDENTIFIED BY '{self.admin_password}';
GRANT ALL ON `{self._name}`.* TO '{self.admin_username}'@'%';

CREATE USER '{self.user_username}'@'%' IDENTIFIED BY '{self.user_password}';
GRANT SELECT, INSERT, UPDATE, DELETE ON `{self._name}`.* TO '{self.user_username}'@'%';
'''

        admin_secret = f'''\
{{
  "host": "10.80.0.3",
  "port": 3306,
  "user": "{self.admin_username}",
  "password": "{self.admin_password}",
  "instance": "db-gh0um",
  "connection_name": "hail-vdc:us-central1:db-gh0um",
  "db": "{self._name}"
}}
'''

        user_secret = f'''\
{{
  "host": "10.80.0.3",
  "port": 3306,
  "user": "{self.user_username}",
  "password": "{self.user_password}",
  "instance": "db-gh0um",
  "connection_name": "hail-vdc:us-central1:db-gh0um",
  "db": "{self._name}"
}}
'''

        script = f'''
set -ex

echo "$SQL_SCRIPT" | mysql --host=10.80.0.3 -u root
kubectl -n {self.namespace} create secret generic {self.admin_secret} --from-literal=sql-config.json="$ADMIN_SECRET"
kubectl -n {self.namespace} create secret generic {self.user_secret} --from-literal=sql-config.json="$USER_SECRET"
'''

        self.job = await batch.create_job(f'gcr.io/{GCP_PROJECT}/ci-utils',
                                          command=['bash', '-c', script],
                                          env={
                                              'SQL_SCRIPT': sql_script,
                                              'ADMIN_SECRET': admin_secret,
                                              'USER_SECRET': user_secret
                                          },
                                          # FIXME configuration
                                          service_account_name='ci2-agent',
                                          parent_ids=self.deps_parent_ids())
