class closing:
    """Async equivalent of :class:`.contextlib.closing`.

    Code like this:

        with closing(await fs.create("gs://abc/123")) as obj:
            <block>

    is equivalent to this:

        f = await fs.create("gs://abc/123")
        try:
            <block>
        finally:
            await f.close()

    """
    def __init__(self, thing):
        self.thing = thing

    async def __aenter__(self):
        return self.thing

    async def __aexit__(self, *exc_info):
        await self.thing.close()
