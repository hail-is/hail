import os
import asyncio
import aiohttp
import unittest
from hailtop.batch_client.aioclient import BatchClient


def async_to_blocking(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class Test(unittest.TestCase):
    def setUp(self):
        session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
        self.client = async_to_blocking(BatchClient(session))

    def tearDown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.close())

    def test_job(self):
        async def f():
            b = self.client.create_batch()
            j = b.create_job('alpine', ['echo', 'test'])
            await b.submit()
            status = await j.wait()
            self.assertTrue('attributes' not in status, (status, await j.log()))
            self.assertEqual(status['state'], 'Success', (status, await j.log()))
            self.assertEqual(status['exit_code']['main'], 0, (status, await j.log()))

            self.assertEqual((await j.log())['main'], 'test\n')

            self.assertTrue(await j.is_complete())

        async_to_blocking(f())
