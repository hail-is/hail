import random
import math
import collections
from hailtop.batch_client.client import BatchClient
import json
import os
import pkg_resources
import secrets
import time
import unittest
import aiohttp
from flask import Flask, Response, request
import requests

import hailtop.gear.auth as hj

from .serverthread import ServerThread


def poll_until(p, max_polls=None):
    i = 0
    while True and (max_polls is None or i < max_polls):
        x = p()
        if x:
            return x
        # max 4.5s
        j = random.randrange(math.floor(1.1 ** min(i, 40)))
        time.sleep(0.100 * j)
        i = i + 1
    raise ValueError(f'poll_until: exceeded max polls: {i} {max_polls}')


class Test(unittest.TestCase):
    def setUp(self):
        session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
        self.client = BatchClient(session, url=os.environ.get('BATCH_URL'))

    def tearDown(self):
        self.client.close()

    def test_job(self):
        builder = self.client.create_batch()
        j = builder.create_job('alpine', ['echo', 'test'])
        builder.submit()
        status = j.wait()
        self.assertTrue('attributes' not in status, (status, j.log()))
        self.assertEqual(status['state'], 'Success', (status, j.log()))
        self.assertEqual(status['exit_code']['main'], 0, (status, j.log()))

        self.assertEqual(j.log()['main'], 'test\n', status)
        j.pod_status()

        self.assertTrue(j.is_complete())

    def test_attributes(self):
        a = {
            'name': 'test_attributes',
            'foo': 'bar'
        }
        builder = self.client.create_batch()
        j = builder.create_job('alpine', ['true'], attributes=a)
        builder.submit()
        status = j.status()
        assert(status['attributes'] == a)

    def test_unsubmitted_state(self):
        builder = self.client.create_batch()
        j = builder.create_job('alpine', ['echo', 'test'])

        with self.assertRaises(ValueError):
            j.batch_id
        with self.assertRaises(ValueError):
            j.id
        with self.assertRaises(ValueError):
            j.status()
        with self.assertRaises(ValueError):
            j.is_complete()
        with self.assertRaises(ValueError):
            j.log()
        with self.assertRaises(ValueError):
            j.pod_status()
        with self.assertRaises(ValueError):
            j.wait()

        builder.submit()
        with self.assertRaises(ValueError):
            builder.create_job('alpine', ['echo', 'test'])

    def test_list_batches(self):
        tag = secrets.token_urlsafe(64)
        b1 = self.client.create_batch(attributes={'tag': tag, 'name': 'b1'})
        b1.create_job('alpine', ['sleep', '3600'])
        b1 = b1.submit()

        b2 = self.client.create_batch(attributes={'tag': tag, 'name': 'b2'})
        b2.create_job('alpine', ['echo', 'test'])
        b2 = b2.submit()

        def assert_batch_ids(expected, complete=None, success=None, attributes=None):
            batches = self.client.list_batches(complete=complete, success=success, attributes=attributes)
            # list_batches returns all batches for all prev run tests
            actual = set([b.id for b in batches]).intersection({b1.id, b2.id})
            self.assertEqual(actual, expected)

        assert_batch_ids({b1.id, b2.id}, attributes={'tag': tag})

        b2.wait()

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

    def test_limit_offset(self):
        b1 = self.client.create_batch()
        for i in range(3):
            b1.create_job('alpine', ['true'])
        b1 = b1.submit()
        s = b1.status(limit=2, offset=1)
        filtered_jobs = {j['job_id'] for j in s['jobs']}
        assert filtered_jobs == {2, 3}, s

    def test_fail(self):
        b = self.client.create_batch()
        j = b.create_job('alpine', ['false'])
        b.submit()
        status = j.wait()
        self.assertEqual(status['exit_code']['main'], 1)

    def test_deleted_job_log(self):
        b = self.client.create_batch()
        j = b.create_job('alpine', ['echo', 'test'])
        b = b.submit()
        j.wait()
        b.delete()

        try:
            j.log()
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                pass
            else:
                self.assertTrue(False, f"batch should have deleted log {e}")

    def test_delete_batch(self):
        b = self.client.create_batch()
        j = b.create_job('alpine', ['sleep', '30'])
        b = b.submit()
        b.delete()

        # verify doesn't exist
        try:
            self.client.get_job(*j.id)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                pass
            else:
                raise

    def test_cancel_batch(self):
        b = self.client.create_batch()
        j = b.create_job('alpine', ['sleep', '30'])
        b = b.submit()

        status = j.status()
        assert status['state'] in ('Ready', 'Running'), status

        b.cancel()

        status = j.wait()
        assert status['state'] == 'Cancelled', status
        assert 'log' not in status, status

        # cancelled job has no log
        try:
            j.log()
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                pass
            else:
                raise

    def test_get_nonexistent_job(self):
        try:
            self.client.get_job(1, 666)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                pass
            else:
                raise

    def test_get_job(self):
        b = self.client.create_batch()
        j = b.create_job('alpine', ['true'])
        b.submit()

        j2 = self.client.get_job(*j.id)
        status2 = j2.status()
        assert (status2['batch_id'], status2['job_id']) == j.id

    def test_batch(self):
        b = self.client.create_batch()
        j1 = b.create_job('alpine', ['false'])
        j2 = b.create_job('alpine', ['sleep', '1'])
        j3 = b.create_job('alpine', ['sleep', '30'])
        b = b.submit()

        j1.wait()
        j2.wait()
        b.cancel()
        bstatus = b.wait()

        assert len(bstatus['jobs']) == 3, bstatus
        state_count = collections.Counter([j['state'] for j in bstatus['jobs']])
        n_cancelled = state_count['Cancelled']
        n_complete = state_count['Error'] + state_count['Failed'] + state_count['Success']
        assert n_cancelled <= 1, bstatus
        assert n_cancelled + n_complete == 3, bstatus

        n_failed = sum([j['exit_code']['main'] > 0 for j in bstatus['jobs'] if j['state'] in ('Failed', 'Error')])
        assert n_failed == 1, bstatus

    def test_batch_status(self):
        b1 = self.client.create_batch()
        b1.create_job('alpine', ['true'])
        b1 = b1.submit()
        b1.wait()
        b1s = b1.status()
        assert b1s['complete'] and b1s['state'] == 'success', b1s

        b2 = self.client.create_batch()
        b2.create_job('alpine', ['false'])
        b2.create_job('alpine', ['true'])
        b2 = b2.submit()
        b2.wait()
        b2s = b2.status()
        assert b2s['complete'] and b2s['state'] == 'failure', b2s

        b3 = self.client.create_batch()
        b3.create_job('alpine', ['sleep', '30'])
        b3 = b3.submit()
        b3s = b3.status()
        assert not b3s['complete'] and b3s['state'] == 'running', b3s

        b4 = self.client.create_batch()
        b4.create_job('alpine', ['sleep', '30'])
        b4 = b4.submit()
        b4.cancel()
        b4.wait()
        b4s = b4.status()
        assert b4s['complete'] and b4s['state'] == 'cancelled', b4s

    def test_callback(self):
        app = Flask('test-client')

        d = {}

        @app.route('/test', methods=['POST'])
        def test():
            body = request.get_json()
            print(f'body {body}')
            d['status'] = body
            return Response(status=200)

        server = ServerThread(app)
        try:
            print('1starting...')
            server.start()
            b = self.client.create_batch()
            j = b.create_job(
                'alpine',
                ['echo', 'test'],
                attributes={'foo': 'bar'},
                callback=server.url_for('/test'))
            b = b.submit()
            print(f'1ids {j.job_id}')
            j.wait()

            poll_until(lambda: 'status' in d)
            status = d['status']
            self.assertEqual(status['state'], 'Success')
            self.assertEqual(status['attributes'], {'foo': 'bar'})
        finally:
            print(f'1shutting down...')
            server.shutdown()
            server.join()
            print(f'1shut down, joined')

    def test_log_after_failing_job(self):
        b = self.client.create_batch()
        j = b.create_job('alpine', ['/bin/sh', '-c', 'echo test; exit 127'])
        b.submit()
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Failed')
        self.assertEqual(status['exit_code']['main'], 127)

        self.assertEqual(j.log()['main'], 'test\n')

        self.assertTrue(j.is_complete())

    def test_authorized_users_only(self):
        endpoints = [
            (requests.get, '/api/v1alpha/batches/0/jobs/0'),
            (requests.get, '/api/v1alpha/batches/0/jobs/0/log'),
            (requests.get, '/api/v1alpha/batches/0/jobs/0/pod_status'),
            (requests.get, '/api/v1alpha/batches'),
            (requests.post, '/api/v1alpha/batches/create'),
            (requests.post, '/api/v1alpha/batches/0/jobs/create'),
            (requests.get, '/api/v1alpha/batches/0'),
            (requests.delete, '/api/v1alpha/batches/0'),
            (requests.patch, '/api/v1alpha/batches/0/close'),
            (requests.get, '/batches'),
            (requests.get, '/batches/0'),
            (requests.get, '/batches/0/jobs/0/log')]
        for f, url in endpoints:
            r = f(os.environ.get('BATCH_URL')+url)
            assert r.status_code == 401, r

    def test_bad_jwt_key(self):
        fname = pkg_resources.resource_filename(
            __name__,
            'jwt-test-user.json')
        with open(fname) as f:
            userdata = json.loads(f.read())
        token = hj.JWTClient(hj.JWTClient.generate_key()).encode(userdata)
        session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
        bc = BatchClient(session, url=os.environ.get('BATCH_URL'), token=token)
        try:
            b = bc.create_batch()
            j = b.create_job('alpine', ['false'])
            b.submit()
            assert False, j
        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                pass
            else:
                assert False, e
        finally:
            bc.close()

    def test_ui_batches(self):
        with open(os.environ['HAIL_TOKEN_FILE']) as f:
            token = f.read()
        # just check successful response
        r = requests.get(f'{os.environ.get("BATCH_URL")}/batches',
                         cookies={'user': token})
        assert (r.status_code >= 200) and (r.status_code < 300)

    def test_ui_batch_and_job_log(self):
        b = self.client.create_batch()
        j = b.create_job('alpine', ['true'])
        b = b.submit()
        status = j.wait()

        with open(os.environ['HAIL_TOKEN_FILE']) as f:
            token = f.read()

        # just check successful response
        r = requests.get(f'{os.environ.get("BATCH_URL")}/batches/{b.id}',
                         cookies={'user': token})
        assert (r.status_code >= 200) and (r.status_code < 300)

        # just check successful response
        r = requests.get(f'{os.environ.get("BATCH_URL")}/batches/{j.batch_id}/jobs/{j.job_id}/log',
                         cookies={'user': token})
        assert (r.status_code >= 200) and (r.status_code < 300)

        r = requests.get(f'{os.environ.get("BATCH_URL")}/batches/{j.batch_id}/jobs/{j.job_id}/pod_status',
                         cookies={'user': token})
        assert (r.status_code >= 200) and (r.status_code < 300)
