import os
import time
import re
import unittest
import batch
from flask import Flask, Response, request
import requests

from .serverthread import ServerThread


class Test(unittest.TestCase):
    def setUp(self):
        self.batch = batch.client.BatchClient(url=os.environ.get('BATCH_URL'))

    def test_job(self):
        j = self.batch.create_job('alpine', ['echo', 'test'])
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 0)

        self.assertEqual(status['log']['main'], 'test\n')
        self.assertEqual(j.log(), {'main': 'test\n'})

        self.assertTrue(j.is_complete())

    def test_create_fails_for_closed_batch(self):
        b = self.batch.create_batch()
        b.close()
        try:
            b.create_job('alpine', ['echo', 'test'])
        except requests.exceptions.HTTPError as err:
            assert err.response.status_code == 400
            assert re.search('.*invalid request: batch_id [0-9]+ is closed', err.response.text)
            return
        assert False

    def test_batch_ttl(self):
        b = self.batch.create_batch(ttl=1)
        t = 1
        while b.status()['is_open']:
            if t > 64:
                assert False, "took more than 128 seconds to close a batch with ttl 1"
            time.sleep(t)
            t = t * 2

    def test_attributes(self):
        a = {
            'name': 'test_attributes',
            'foo': 'bar'
        }
        j = self.batch.create_job('alpine', ['true'], attributes=a)
        status = j.status()
        assert(status['attributes'] == a)

    def test_scratch_folder(self):
        sb = 'gs://test-bucket/folder'
        j = self.batch.create_job('alpine', ['true'], scratch_folder=sb)
        status = j.status()
        assert(status['scratch_folder'] == sb)

    def test_fail(self):
        j = self.batch.create_job('alpine', ['false'])
        status = j.wait()
        self.assertEqual(status['exit_code'], 1)

    def test_deleted_job_log(self):
        j = self.batch.create_job('alpine', ['echo', 'test'])
        id = j.id
        j.wait()
        j.delete()
        self.assertEqual(self.batch._get_job_log(id), {'main': 'test\n'})

    def test_delete_job(self):
        j = self.batch.create_job('alpine', ['sleep', '30'])
        id = j.id
        j.delete()

        # verify doesn't exist
        try:
            self.batch._get_job(id)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

    def test_cancel_job(self):
        j = self.batch.create_job('alpine', ['sleep', '30'])
        status = j.status()
        self.assertTrue(status['state'], 'Created')

        j.cancel()

        status = j.status()
        self.assertTrue(status['state'], 'Cancelled')
        self.assertTrue('log' not in status)

        # cancelled job has no log
        try:
            j.log()
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

    def test_get_nonexistent_job(self):
        try:
            self.batch._get_job(666)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

    def test_api_cancel_nonexistent_job(self):
        try:
            self.batch._cancel_job(666)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

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

        # test list_jobs
        jobs = self.batch.list_jobs()
        self.assertTrue(
            set([j.id for j in jobs]).issuperset([j1.id, j2.id, j3.id]))

        # test refresh_k8s_state
        self.batch._refresh_k8s_state()

        j2.wait()
        j3.cancel()
        bstatus = b.wait()

        n_cancelled = bstatus['jobs']['Cancelled']
        n_complete = bstatus['jobs']['Complete']
        self.assertTrue(n_cancelled <= 1)
        self.assertTrue(n_cancelled + n_complete == 3)

        n_failed = sum([ec > 0 for _, ec in bstatus['exit_codes'].items() if ec is not None])
        self.assertTrue(n_failed == 1)

    def test_callback(self):
        app = Flask('test-client')

        d = {}

        @app.route('/test', methods=['POST'])
        def test():
            d['status'] = request.get_json()
            return Response(status=200)

        server = ServerThread(app)
        try:
            server.start()

            j = self.batch.create_job(
                'alpine',
                ['echo', 'test'],
                attributes={'foo': 'bar'},
                callback=server.url_for('/test'))
            j.wait()

            status = d['status']
            self.assertEqual(status['state'], 'Complete')
            self.assertEqual(status['attributes'], {'foo': 'bar'})
        finally:
            server.shutdown()
            server.join()

    def test_inputs(self):
        j = self.batch.create_job('alpine', ['echo', 'main'], input_files=['foo'])
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 0)

        self.assertEqual(status['log'], {'main': 'main\n', 'input': 'hello\n'})
        self.assertEqual(j.log(), {'main': 'main\n', 'input': 'hello\n'})

        self.assertTrue(j.is_complete())

    def test_outputs(self):
        j = self.batch.create_job('alpine', ['echo', 'main'], output_files=['foo'])
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 0)

        self.assertEqual(status['log'], {'main': 'main\n', 'output': 'hello\n'})
        self.assertEqual(j.log(), {'main': 'main\n', 'output': 'hello\n'})

        self.assertTrue(j.is_complete())

    def test_inputs_and_outputs(self):
        j = self.batch.create_job('alpine', ['echo', 'main'],
                                  input_files=['foo_in'], output_files=['foo_out'])
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 0)

        self.assertEqual(status['log'], {'main': 'main\n',
                                         'input': 'hello\n',
                                         'output': 'hello\n'})
        self.assertEqual(j.log(), {'main': 'main\n',
                                   'input': 'hello\n',
                                   'output': 'hello\n'})

        self.assertTrue(j.is_complete())

    def test_inputs_and_outputs_delete(self):
        j = self.batch.create_job('alpine', ['sleep', '30'],
                                  input_files=['foo_in'], output_files=['foo_out'])
        id = j.id
        j.delete()

        # verify doesn't exist
        try:
            self.batch._get_job(id)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

    def test_inputs_outputs_cancel(self):
        j = self.batch.create_job('alpine', ['sleep', '30'],
                                  input_files=['foo_in'], output_files=['foo_out'])
        status = j.status()
        self.assertTrue(status['state'], 'Created')

        j.cancel()

        status = j.status()
        self.assertTrue(status['state'], 'Cancelled')
        self.assertTrue('log' not in status)

        # cancelled job has no log
        try:
            j.log()
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

    def test_log_after_failing_job(self):
        j = self.batch.create_job('alpine', ['/bin/sh', '-c', 'echo test; exit 127'])
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 127)

        self.assertEqual(status['log']['main'], 'test\n')
        self.assertEqual(j.log(), {'main': 'test\n'})

        self.assertTrue(j.is_complete())
