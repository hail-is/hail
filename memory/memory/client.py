from hailtop.aiogoogle.client.storage_client import GoogleStorageAsyncFS


class MemoryClient:
    def __init__(self, gcs_project=None, fs=None):
        if fs is None:
            fs = GoogleStorageAsyncFS(project=gcs_project)
        self._fs = fs

    async def read_file(self, filename):
        if self._fs.isfile(filename):
            async with await self._fs.open(filename) as f:
                return await f.read()
        return None

    async def write_file(self, filename, data):
        async with await self._fs.create(filename) as f:
            await f.write(data)

    async def close(self):
        await self._fs.close()
