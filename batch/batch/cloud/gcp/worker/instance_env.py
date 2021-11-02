import os

from ....worker.instance_env import CloudInstanceEnvironment


class GCPInstanceEnvironment(CloudInstanceEnvironment):
    @staticmethod
    def from_env():
        project = os.environ['PROJECT']
        zone = os.environ['ZONE'].rsplit('/', 1)[1]
        return GCPInstanceEnvironment(project, zone)

    def __init__(self, project: str, zone: str):
        self.project = project
        self.zone = zone

    def __str__(self):
        return f'project={self.project} zone={self.zone}'
