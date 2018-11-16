from .job import Job
from ..log import log
from .. import data
from .user_error import UserError


class DagNode:
    def __init__(self, spec, parents, children):
        self.spec = spec
        self._parents = parents
        self.children = children
        self.unfinished_parents = set([])
        self.job = None

    def add_parent(self, parent):
        self._parents.append(parent)
        self.unfinished_parents.add(parent.spec.name)

    def add_child(self, child):
        self.children.append(child)

    def to_json(self):
        return {
            'spec': self.spec.to_json(),
            'parents': [parent.spec.name for parent in self._parents],
            'children': [child.spec.name for child in self.children]
        }

    def __str__(self):
        return str(self.to_json())

    def __repr__(self):
        return self.__str__()

    def to_get_json(self):
        return {
            'job': self.job,
            'parents': [parent.name for parent in self._parents],
            'children': [child.name for child in self.children]
        }

    def link_job(self, job):
        self.job = job

    def mark_complete(self):
        exit_code = self.job.exit_code
        if exit_code == 0:
            for child in self.children:
                child.mark_parent_complete(self)
        else:
            log.warning(f'failing pod {exit_code} will not trigger dag completion')

    def mark_parent_complete(self, parent):
        parent_name = parent.spec.name
        log.info(f'parent {parent_name}/{parent.job.id} completed for child {self.spec.name}')
        if parent_name not in self.unfinished_parents:
            raise ValueError(
                f'unknown parent: {parent_name} not in {self.unfinished_parents}. '
                f'{self}')
        self.unfinished_parents.remove(parent_name)
        if not self.unfinished_parents:
            log.info(f'all parents completed for child {self.spec.name}')
            assert self.job is None
            self.create_job()
            assert self.job is not None

    def create_job(self):
        job_spec = self.spec.job_spec
        return Job(job_spec.pod_spec, None, job_spec.attributes, job_spec.callback, self)


class Dag:
    schema = data.Dag.schema
    validator = data.Dag.validator

    @staticmethod
    def from_json(id, doc):
        return Dag(id, data.Dag.from_json(doc).nodes)

    def __init__(self, id, specs):
        self.id = id
        self.roots = []
        self.nodes = []

        by_name = {}
        for spec in specs:
            node = DagNode(spec, [], [])
            if spec.name in by_name:
                raise UserError(f'duplicate name: {spec.name}')
            by_name[spec.name] = node
            self.nodes.append(node)
        for child in by_name.values():
            if not child.spec.parent_names:
                self.roots.append(child)
            else:
                for parent_name in child.spec.parent_names:
                    parent = by_name.get(parent_name, None)
                    if not parent:
                        raise UserError(f'parent not found: {parent_name}, {by_name.keys()}')
                    parent.add_child(child)
                    child.add_parent(parent)
        for root in self.roots:
            root.create_job()

    def to_data_dag(self):
        return data.Dag([node.spec for node in self.nodes])

    def to_get_json(self):
        return self.to_data_dag().to_json()
