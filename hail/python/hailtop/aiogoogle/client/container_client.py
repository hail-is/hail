from .base_client import BaseClient


class ContainerClient(BaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://gcr.io/v2/{project}', **kwargs)

    # list images:
    # GET /tags/list
    # returns:
    # {
    #   'child': [<image>, ...],
    #   'manifest': {},
    #   'name': <project>,
    #   'tags': []
    # }

    # list image tags:
    # GET /{image}/tags/list
    # returns:
    # {
    #   'child': [],
    #   'manifest': {<diget>: {
    #       'imageSizeBytes': <size>,
    #       'layerId': '',
    #       'mediaType': 'application/vnd.docker.distribution.manifest.v2+json',
    #       'tag': [<tag>, ...],
    #       'timeCreatedMs': '<time-in-ms>',
    #       'timeUploadedMs': '<time-in-ms>'
    #     }, ...
    #   },
    #   'name': '<project>/<image>',
    #   'tags': [<tag>, ...]
    # }

    # delete image tag
    # DELETE /{image}/manifests/{tag}

    # delete image digest
    # DELETE /{image}/manifests/{digest}
