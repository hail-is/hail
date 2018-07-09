import threading
import time
import os
import unittest
import batch
import requests
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

    def test_callback(self):
        app = Flask('test-client')

        d = {}

        @app.route('/test', methods=['POST'])
        def test():
            d['status'] = request.get_json()
            return Response(status=200)

        server = ServerThread(app, host=self.ip, port=5869)
        server.start()

        j = self.batch.create_job('alpine', ['echo', 'test'],
                                  attributes={'foo': 'bar'},
                                  callback='http://{}:5869/test'.format(self.ip))
        j.wait()

        status = d['status']
        self.assertEqual(status['state'], 'Complete')
        self.assertEqual(status['attributes'], {'foo': 'bar'})

        server.shutdown()
        server.join()
