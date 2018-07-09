import os
import unittest
import batch

class Test(unittest.TestCase):
    def setUp(self):
        self.batch = batch.client.BatchClient(
            url = os.environ.get('BATCH_URL'))

    def test_job(self):
        j = self.batch.create_job('alpine', ['echo', 'test'])
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 0)
        self.assertEqual(status['log'], 'test\n')
        self.assertTrue(j.is_complete())

    def test_attributes(self):
        a = {
            'name': 'test_attributes',
            'foo': 'bar'
        }
        j = self.batch.create_job(
            'alpine', ['true'],
            attributes = a)
        status = j.status()
        print(status)
        assert(status['attributes'] == a)

    def test_fail(self):
        j = self.batch.create_job('alpine', ['false'])
        status = j.wait()
        self.assertEqual(status['exit_code'], 1)

    def test_cancel_job(self):
        j = self.batch.create_job('alpine', ['sleep', '30'])
        status = j.status()
        self.assertTrue(status['state'], 'Created')

        j.cancel()
        status = j.status()
        self.assertTrue(status['state'], 'Cancelled')

    def test_get_job(self):
        j = self.batch.create_job('alpine', ['true'])
        j2 = self.batch.get_job(j.id)
        status2 = j2.status()
        assert(status2['id'] == j.id)

    def test_batch(self):
        b = self.batch.create_batch()
        j1 = b.create_job('alpine', ['false'])
        j2 = b.create_job('alpine', ['sleep', '1'])
        j3 = b.create_job('alpine', ['sleep', '30'])
        j2.wait()
        j3.cancel()
        bstatus = b.wait()
        self.assertTrue(bstatus['jobs']['Cancelled'] == 1)
        self.assertTrue(bstatus['jobs']['Complete'] == 2)
