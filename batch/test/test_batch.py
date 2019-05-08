
import collections
import batch
import json
import os
import pkg_resources
import re
import secrets
import time
import unittest
from flask import Flask, Response, request
import requests

import hailjwt as hj

from .serverthread import ServerThread


class Test(unittest.TestCase):
    def setUp(self):
        self.batch = batch.client.BatchClient(url=os.environ.get('BATCH_URL'))

    def test_job(self):
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['echo', 'test'])
        b.close()
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 0)

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
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['true'], attributes=a)
        status = j.status()
        assert(status['attributes'] == a)

    def test_list_batches(self):
        tag = secrets.token_urlsafe(64)
        b1 = self.batch.create_batch(attributes={'tag': tag, 'name': 'b1'})
        b1.create_job('alpine', ['sleep', '30'])
        b1.close()

        b2 = self.batch.create_batch(attributes={'tag': tag, 'name': 'b2'})
        b2.create_job('alpine', ['echo', 'test'])
        b2.close()

        def assert_batch_ids(expected, complete=None, success=None, attributes=None):
            batches = self.batch.list_batches(complete=complete, success=success, attributes=attributes)
            # list_batches returns all batches for all prev run tests
            actual = set([batch.id for batch in batches]).intersection({b1.id, b2.id})
            print(actual, expected)
            self.assertEqual(actual, expected)

        assert_batch_ids({b1.id, b2.id}, attributes={'tag': tag})

        s = b2.wait()
        print(b1.status())
        print(s)
        assert_batch_ids({b1.id}, complete=False, attributes={'tag': tag})
        assert_batch_ids({b2.id}, complete=True, attributes={'tag': tag})

        assert_batch_ids({b1.id}, success=False, attributes={'tag': tag})
        assert_batch_ids({b2.id}, success=True, attributes={'tag': tag})

        b1.cancel()
        b1.wait()

        assert_batch_ids({b1.id}, success=False, attributes={'tag': tag})
        assert_batch_ids({b2.id}, success=True, attributes={'tag': tag})

        assert_batch_ids(set(), complete=False, attributes={'tag': tag})
        assert_batch_ids({b1.id, b2.id}, complete=True, attributes={'tag': tag})

        assert_batch_ids({b2.id}, attributes={'tag': tag, 'name': 'b2'})

    def test_fail(self):
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['false'])
        b.close()
        status = j.wait()
        self.assertEqual(status['exit_code'], 1)

    def test_deleted_job_log(self):
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['echo', 'test'])
        b.close()
        id = j.id
        j.wait()
        b.delete()

        try:
            self.batch._get_job_log(id)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                self.assertTrue(False, f"batch should not have deleted log {e}")

    def test_delete_batch(self):
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['sleep', '30'])
        b.close()
        id = j.id
        b.delete()

        # verify doesn't exist
        try:
            self.batch._get_job(id)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

    def test_cancel_batch(self):
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['sleep', '30'])
        b.close()

        status = j.status()
        self.assertTrue(status['state'], 'Ready')

        b.cancel()

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

    def test_get_job(self):
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['true'])
        b.close()

        j2 = self.batch.get_job(j.id)
        status2 = j2.status()
        assert(status2['id'] == j.id)

    def test_batch(self):
        b = self.batch.create_batch()
        j1 = b.create_job('alpine', ['false'])
        j2 = b.create_job('alpine', ['sleep', '1'])
        j3 = b.create_job('alpine', ['sleep', '30'])
        b.close()

        j1.wait()
        j2.wait()
        b.cancel()
        bstatus = b.wait()

        assert(len(bstatus['jobs']) == 3)
        state_count = collections.Counter([j['state'] for j in bstatus['jobs']])
        n_cancelled = state_count['Cancelled']
        n_complete = state_count['Complete']
        self.assertTrue(n_cancelled <= 1)
        self.assertTrue(n_cancelled + n_complete == 3)

        n_failed = sum([j['exit_code'] > 0 for j in bstatus['jobs'] if j['state'] == 'Complete'])
        self.assertTrue(n_failed == 1)

    def test_batch_status(self):
        b1 = self.batch.create_batch()
        b1.create_job('alpine', ['true'])
        b1.close()
        b1.wait()
        b1s = b1.status()
        assert b1s['complete'] and b1s['state'] == 'success', b1s

        b2 = self.batch.create_batch()
        b2.create_job('alpine', ['false'])
        b2.create_job('alpine', ['true'])
        b2.close()
        b2.wait()
        b2s = b2.status()
        assert b2s['complete'] and b2s['state'] == 'failure', b2s

        b3 = self.batch.create_batch()
        b3.create_job('alpine', ['sleep', '30'])
        b3.close()
        b3s = b3.status()
        assert not b3s['complete'] and b3s['state'] == 'running', b3s

        b4 = self.batch.create_batch()
        b4.create_job('alpine', ['sleep', '30'])
        b4.close()
        b4.cancel()
        b4.wait()
        b4s = b4.status()
        assert b4s['complete'] and b4s['state'] == 'cancelled', b4s

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
            b = self.batch.create_batch()
            j = b.create_job(
                'alpine',
                ['echo', 'test'],
                attributes={'foo': 'bar'},
                callback=server.url_for('/test'))
            b.close()
            j.wait()

            batch.poll_until(lambda: 'status' in d)
            status = d['status']
            self.assertEqual(status['state'], 'Complete')
            self.assertEqual(status['attributes'], {'foo': 'bar'})
        finally:
            server.shutdown()
            server.join()

    def test_log_after_failing_job(self):
        b = self.batch.create_batch()
        j = b.create_job('alpine', ['/bin/sh', '-c', 'echo test; exit 127'])
        b.close()
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 127)

        self.assertEqual(j.log(), {'main': 'test\n'})

        self.assertTrue(j.is_complete())

    def test_authorized_users_only(self):
        endpoints = [
            (requests.post, '/jobs/create'),
            (requests.get, '/jobs/0'),
            (requests.get, '/jobs/0/log'),
            (requests.get, '/batches'),
            (requests.post, '/batches/create'),
            (requests.get, '/batches/0'),
            (requests.delete, '/batches/0'),
            (requests.patch, '/batches/0/close')]
        for f, url in endpoints:
            r = f(os.environ.get('BATCH_URL')+url)
            assert r.status_code == 401, r

    def test_bad_jwt_key(self):
        fname = pkg_resources.resource_filename(
            __name__,
            'jwt-test-user.json')
        with open(fname) as f:
            userdata = json.loads(f.read())
        token = hj.JWTClient(hj.JWTClient.generate_key()).encode(userdata).decode('ascii')
        bc = batch.client.BatchClient(url=os.environ.get('BATCH_URL'), token=token)
        try:
            b = bc.create_batch()
            j = b.create_job('alpine', ['false'])
            b.close()
            assert False, j
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                pass
            else:
                assert False, e
