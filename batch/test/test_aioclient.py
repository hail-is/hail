import os
import asyncio
import unittest
import batch

class Test(unittest.TestCase):
    def setUp(self):
        self.client = batch.aioclient.BatchClient(url=os.environ.get('BATCH_URL'))

    def tearDown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.close())

    def test_job(self):
        async def f():
            j = await self.client.create_job('alpine', ['echo', 'test'])
            status = await j.wait()
            self.assertTrue('attributes' not in status)
            self.assertEqual(status['state'], 'Complete')
            self.assertEqual(status['exit_code'], 0)

            self.assertEqual(status['log']['main'], 'test\n')
            self.assertEqual(await j.log(), {'main': 'test\n'})

            self.assertTrue(await j.is_complete())

        loop = asyncio.get_event_loop()
        loop.run_until_complete(f())
