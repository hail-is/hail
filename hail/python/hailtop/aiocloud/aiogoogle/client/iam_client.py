from .base_client import GCPBaseClient


class GoogleIAmClient(GCPBaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://iam.googleapis.com/v1/projects/{project}', **kwargs)

    # https://cloud.google.com/iam/docs/reference/rest
