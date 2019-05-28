import os
import asyncio
import aiohttp
import unittest
import batch


class Test(unittest.TestCase):
    def setUp(self):
        self.session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
        self.client = batch.client.AsyncBatchClient(self.session, url=os.environ.get('BATCH_URL'))

    def tearDown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.session.close())

    def test_job(self):
        async def f():
            b = await self.client.create_batch()
            j = await b.create_job('alpine', ['echo', 'test'])
            status = await j.wait()
            self.assertTrue('attributes' not in status)
            self.assertEqual(status['state'], 'Complete')
            self.assertEqual(status['exit_code']['main'], 0)

            self.assertEqual(await j.log(), {'main': 'test\n'})

            self.assertTrue(await j.is_complete())

        loop = asyncio.get_event_loop()
        loop.run_until_complete(f())
