from .base_client import BaseClient


class ComputeClient(BaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://compute.googleapis.com/compute/v1/projects/{project}', **kwargs)

    # docs:
    # https://cloud.google.com/compute/docs/reference/rest/v1
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/insert
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/get
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/delete
