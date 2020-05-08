import hailtop.aiogoogle.auth as google_auth


class Client:
    def __init__(self, project, *, session=None, **kwargs):
        self._project = project
        if session is None:
            session = google_auth.Session(**kwargs)
        self._session = session

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

    async def get(self, path, **kwargs):
        async with await self._session.get(
                f'https://gcr.io/v2/{self._project}{path}', **kwargs) as resp:
            return await resp.json()

    # delete image tag
    # DELETE /{image}/manifests/{tag}

    # delete image digest
    # DELETE /{image}/manifests/{digest}

    async def delete(self, path, **kwargs):
        async with await self._session.delete(
                f'https://gcr.io/v2/{self._project}{path}', **kwargs):
            pass

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
