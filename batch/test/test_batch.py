import batch
import os
import re
import requests
import threading
import time
import unittest
from werkzeug.serving import make_server
from flask import Flask, request, jsonify, url_for, Response

class ServerThread(threading.Thread):
    def __init__(self, app, host='127.0.0.1', port=5000):
        super().__init__()

        @app.route('/ping', methods=['GET'])
        def ping():
            return Response(status=200)

        self.host = host
        self.port = port
        self.app = app
        self.server = make_server(self.host, self.port, app)
        self.context = app.app_context()
        self.context.push()

    def ping(self):
        ping_url = 'http://{}:{}/ping'.format(self.host, self.port)

        up = False
        while not up:
            try:
                requests.get(ping_url)
                up = True
            except requests.exceptions.ConnectionError:
                time.sleep(0.01)

    def start(self):
        super().start()
        self.ping()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

class Test(unittest.TestCase):
    def setUp(self):
        self.batch = batch.client.BatchClient(
            url = os.environ.get('BATCH_URL'))

        self.ip = os.environ.get('POD_IP')
        if not self.ip:
            self.ip = '127.0.0.1'

    def test_job(self):
        j = self.batch.create_job('alpine', ['echo', 'test'])
        status = j.wait()
        self.assertTrue('attributes' not in status)
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['exit_code'], 0)

        self.assertEqual(status['log'], 'test\n')
        self.assertEqual(j.log(), 'test\n')

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
        assert(status['attributes'] == a)

    def test_fail(self):
        j = self.batch.create_job('alpine', ['false'])
        status = j.wait()
        self.assertEqual(status['exit_code'], 1)

    def test_deleted_job_log(self):
        j = self.batch.create_job('alpine', ['echo', 'test'])
        id = j.id
        j.wait()
        j.delete()
        self.assertEqual(self.batch._get_job_log(id), 'test\n')

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

    def test_callback(self):
        app = Flask('test-client')

        d = {}

        @app.route('/test', methods=['POST'])
        def test():
            d['status'] = request.get_json()
            return Response(status=200)

        port = 5869
        server = ServerThread(app, host=self.ip, port=port)
        try:
            server.start()

            j = self.batch.create_job('alpine', ['echo', 'test'],
                                      attributes={'foo': 'bar'},
                                      callback='http://{}:{}/test'.format(self.ip, port))
            j.wait()

            status = d['status']
            self.assertEqual(status['state'], 'Complete')
            self.assertEqual(status['attributes'], {'foo': 'bar'})
        finally:
            server.shutdown()
            server.join()
