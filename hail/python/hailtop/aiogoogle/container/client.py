import hailtop.aiogoogle.auth as google_auth


class Client:
    def __init__(self, project, *, session=None, **kwargs):
        self._project = project
        if session is None:
            session = google_auth.Session(**kwargs)
        self._session = session

    # returns:
    # {
    #   'child': [<image>, ...],
    #   'manifest': {},
    #   'name': <project>,
    #   'tags': []
    # }
    async def list_images(self, **kwargs):
        async with await self._session.get(
                f'https://gcr.io/v2/{self._project}/tags/list', **kwargs) as resp:
            return await resp.json()

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
    async def list_image_tags(self, image, **kwargs):
        async with await self._session.get(
                f'https://gcr.io/v2/{self._project}/{image}/tags/list', **kwargs) as resp:
            return await resp.json()

    async def delete_image_tag(self, image, tag, **kwargs):
        async with await self._session.delete(
                f'https://gcr.io/v2/{self._project}/{image}/manifests/{tag}', **kwargs):
            pass

    async def delete_image(self, image, digest, **kwargs):
        async with await self._session.delete(
                f'https://gcr.io/v2/{self._project}/{image}/manifests/{digest}', **kwargs):
            pass

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
