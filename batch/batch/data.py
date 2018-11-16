import cerberus
import kubernetes as kube

from .kube_client import kube_api_client


class JobSpec:
    schema = {
        'pod_spec': {
            'type': 'dict',
            'required': True,
            'allow_unknown': True,
            'schema': {}  # checked by k8s
        },
        'batch_id': {'type': 'integer'},
        'attributes': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        },
        'callback': {'type': 'string'}
    }
    validator = cerberus.Validator(schema)

    @staticmethod
    def from_json(doc):
        return JobSpec(
            kube_api_client._ApiClient__deserialize(
                doc['pod_spec'],
                kube.client.V1PodSpec),
            doc.get('attributes', None),
            doc.get('batch_id', None),
            doc.get('callback', None))

    @staticmethod
    def from_parameters(image,
                        command=None,
                        args=None,
                        env=None,
                        ports=None,
                        resources=None,
                        tolerations=None,
                        volumes=None,
                        security_context=None,
                        service_account_name=None,
                        attributes=None,
                        batch_id=None,
                        callback=None):
        ports = [] if ports is None else ports
        volumes = [] if volumes is None else volumes
        if env:
            env = [{'name': k, 'value': v} for (k, v) in env.items()]
        else:
            env = []
        env.extend([{
            'name': 'POD_IP',
            'valueFrom': {
                'fieldRef': {'fieldPath': 'status.podIP'}
            }
        }, {
            'name': 'POD_NAME',
            'valueFrom': {
                'fieldRef': {'fieldPath': 'metadata.name'}
            }
        }])

        pod_spec = kube.client.V1PodSpec(
            containers=[kube.client.V1Container(
                image=image,
                name='default',
                command=command,
                args=args,
                env=env,
                ports=[
                    kube.client.V1ContainerPort(
                        container_port=p,
                        protocol='TCP')
                    for p in ports],
                resources=resources,
                volume_mounts=[v['volume_mount'] for v in volumes])],
            restart_policy='Never',
            service_account_name=service_account_name,
            security_context=security_context,
            tolerations=tolerations,
            volumes=[v['volume'] for v in volumes])

        return JobSpec(pod_spec, attributes, batch_id, callback)

    def __init__(self, pod_spec, attributes, batch_id, callback):
        self.pod_spec = pod_spec
        self.attributes = attributes
        self.batch_id = batch_id
        self.callback = callback

    def to_json(self):
        doc = {'pod_spec': self.pod_spec.to_dict()}
        if self.attributes:
            doc['attributes'] = self.attributes
        if self.batch_id:
            doc['batch_id'] = self.batch_id
        if self.callback:
            doc['callback'] = self.callback
        return doc

    def __str__(self):
        return str(self.to_json())

    def __repr__(self):
        return self.__str__()


class DagNodeSpec:
    schema = {
        'name': {'type': 'string'},
        'parent_names': {'type': 'list', 'schema': {'type': 'string'}},
        'job_spec': {'type': 'dict', 'schema': JobSpec.schema}
    }
    validator = cerberus.Validator(schema)

    @staticmethod
    def from_json(doc):
        try:
            return DagNodeSpec(doc['name'],
                               doc['parent_names'],
                               JobSpec.from_json(doc['job_spec']))
        except KeyError as err:
            raise ValueError(
                f'could not parse {doc} into a DagNodeSpec') from err

    def __init__(self, name, parent_names, job_spec):
        self.name = name
        self.parent_names = parent_names
        self.job_spec = job_spec

    def to_json(self):
        return {
            'name': self.name,
            'parent_names': self.parent_names,
            'job_spec': self.job_spec.to_json(),
        }

    def __str__(self):
        return str(self.to_json())

    def __repr__(self):
        return self.__str__()


class Dag:
    schema = {
        'nodes': {'type': 'list', 'schema': {'type': 'dict', 'schema': DagNodeSpec.schema}}
    }
    validator = cerberus.Validator(schema)

    @staticmethod
    def from_json(doc):
        try:
            return Dag([DagNodeSpec.from_json(node) for node in doc['nodes']])
        except KeyError as err:
            raise ValueError(
                f'could not parse {doc} into a Dag') from err

    def __init__(self, nodes):
        self.nodes = nodes

    def to_json(self):
        return {'nodes': [node.to_json() for node in self.nodes]}

    def __str__(self):
        return str(self.to_json())

    def __repr__(self):
        return self.__str__()
