import sys
import os
import time
import random
from collections import Counter
import logging
import threading
from flask import Flask, request, jsonify, abort, url_for
import kubernetes as kube
import cerberus
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('batch')

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
v1 = kube.client.CoreV1Api()

counter = 0
def next_id():
    global counter

    counter = counter + 1
    return counter

pod_name_job = {}
job_id_job = {}

class Job(object):
    def _create_pod(self):
        assert not self._pod_name

        pod = v1.create_namespaced_pod('default', self.pod_template)
        self._pod_name = pod.metadata.name
        pod_name_job[self._pod_name] = self

        log.info('created pod name: {} for job {}'.format(self._pod_name, self.id))

    def _cancel_pod(self):
        if self._pod_name:
            try:
                v1.delete_namespaced_pod(self._pod_name, 'default', kube.client.V1DeleteOptions())
            except kube.client.rest.ApiException as e:
                if e.status == 404 and e.reason == 'NotFound':
                    pass
                else:
                    raise
            del pod_name_job[self._pod_name]
            self._pod_name = None

    def __init__(self, pod_spec, batch_id, attributes, callback):
        self.id = next_id()
        job_id_job[self.id] = self

        self.batch_id = batch_id
        if batch_id:
            batch = batch_id_batch[batch_id]
            batch.jobs.append(self)

        self.attributes = attributes
        self.callback = callback

        self.pod_template = kube.client.V1Pod(
            metadata = kube.client.V1ObjectMeta(generate_name = 'job-{}-'.format(self.id)),
            spec = pod_spec)

        self._pod_name = None

        self._state = 'Created'
        log.info('created job {}'.format(self.id))

        self._create_pod()

    def set_state(self, new_state):
        if self._state != new_state:
            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))
            self._state = new_state

    def cancel(self):
        if self.is_complete():
            return
        self._cancel_pod()
        self.set_state('Cancelled')

    def is_complete(self):
        return self._state == 'Complete' or self._state == 'Cancelled'

    def mark_unscheduled(self):
        if self._pod_name:
            del pod_name_job[self._pod_name]
            self._pod_name = None
        self._create_pod()

    def mark_complete(self, pod):
        self.exit_code = pod.status.container_statuses[0].state.terminated.exit_code
        self.log = v1.read_namespaced_pod_log(pod.metadata.name, 'default')

        log.info('job {} complete, exit_code {}, log:\n{}'.format(
            self.id, self.exit_code, self.log))
        self.set_state('Complete')

        if self.callback:
            try:
                requests.post(self.callback, json = self.to_json())
            except requests.exceptions.RequestException as re:
                id = self.id
                log.warn(f'callback for job {id} failed due to an error, I will not retry. Error: {re}')

    def to_json(self):
        result = {
            'id': self.id,
            'state': self._state
        }
        if self._state == 'Complete':
            result['exit_code'] = self.exit_code
            result['log'] = self.log
        if self.attributes:
            result['attributes'] = self.attributes
        return result

app = Flask('batch')

@app.route('/jobs/create', methods=['POST'])
def create_job():
    parameters = request.json

    schema = {
        # will be validated when creating pod
        'spec': {'type': 'dict',
                 'required': True,
                 'allow_unknown': True,
                 'schema': {}
        },
        'batch_id': {'type': 'integer'},
        'attributes': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        },
        'callback': {'type': 'string'}
    }
    v = cerberus.Validator(schema)
    if (not v.validate(parameters)):
        # print(v.errors)
        abort(404, 'invalid request: {}'.format(v.errors))

    pod_spec = v1.api_client._ApiClient__deserialize(
        parameters['spec'], kube.client.V1PodSpec)

    batch_id = parameters.get('batch_id')
    if batch_id:
        if batch_id not in batch_id_batch:
            abort(404, 'valid request: batch_id {} not found'.format(batch_id))

    job = Job(
        pod_spec, batch_id, parameters.get('attributes'), parameters.get('callback'))
    return jsonify(job.to_json())

@app.route('/jobs', methods=['GET'])
def get_job_list():
    return jsonify([job.to_json() for _, job in job_id_job.items()])

@app.route('/jobs/<int:job_id>', methods=['GET'])
def get_job(job_id):
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    return jsonify(job.to_json())

@app.route('/jobs/<int:job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    job.cancel()
    return jsonify({})

batch_id_batch = {}

class Batch(object):
    def __init__(self, attributes):
        self.attributes = attributes
        self.id = next_id()
        batch_id_batch[self.id] = self
        self.jobs = []

    def delete(self):
        del batch_id_batch[self.id]
        for j in self.jobs:
            assert j.batch_id == self.id
            j.batch_id = None

    def to_json(self):
        state_count = Counter([j._state for j in self.jobs])
        return {
            'id': self.id,
            'jobs': {
                'Created': state_count.get('Created', 0),
                'Complete': state_count.get('Complete', 0),
                'Cancelled': state_count.get('Cancelled', 0)
            },
            'attributes': self.attributes
        }

@app.route('/batches/create', methods=['POST'])
def create_batch():
    parameters = request.json

    schema = {
        'attributes': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        }
    }
    v = cerberus.Validator(schema)
    if (not v.validate(parameters)):
        abort(404, 'invalid request: {}'.format(v.errors))

    batch = Batch(parameters.get('attributes'))
    return jsonify(batch.to_json())

@app.route('/batches/<int:batch_id>', methods=['GET'])
def get_batch(batch_id):
    batch = batch_id_batch.get(batch_id)
    if not batch:
        abort(404)
    return jsonify(batch.to_json())

@app.route('/batches/<int:batch_id>/delete', methods=['DELETE'])
def delete_batch(batch_id):
    batch = batch_id_batch.get(batch_id)
    if not batch:
        abort(404)
    batch.delete()
    return jsonify({})

def run_forever(target, *args, **kwargs):
    # target should be a function
    target_name = target.__name__

    expected_retry_interval_ms = 15 * 1000 # 15s
    while True:
        start = time.time()
        try:
            log.info(f'run_forever: run target {target_name}')
            target(*args, **kwargs)
            log.info(f'run_forever: target {target_name} returned')
        except:
            log.error(f'run_forever: target {target_name} threw exception', exc_info=sys.exc_info())
        end = time.time()

        run_time_ms = int((end - start) * 1000 + 0.5)
        t = random.randrange(expected_retry_interval_ms * 2) - run_time_ms
        if t > 0:
            log.debug(f'run_forever: {target_name}: sleep {t}ms')
            time.sleep(t / 1000.0)

def flask_event_loop():
    app.run(debug=True, host='0.0.0.0')

def kube_event_loop():
    stream = kube.watch.Watch().stream(v1.list_namespaced_pod, 'default')
    for event in stream:
        event_type = event['type']
        pod = event['object']
        name = pod.metadata.name

        # too much
        # log.debug(f'kube_event_loop: got event {event}')
        log.debug(f'kube_event_loop: got event: {event_type} {pod.api_version}/{pod.kind} {name}')

        job = pod_name_job.get(name)
        if job and not job.is_complete():
            if event_type == 'DELETE':
                job.mark_unscheduled()
            elif event_type == 'ADDED' or event_type == 'MODIFIED':
                if pod.status.container_statuses:
                    assert len(pod.status.container_statuses) == 1
                    container_status = pod.status.container_statuses[0]
                    assert container_status.name == 'default'

                    if container_status.state and container_status.state.terminated:
                        job.mark_complete(pod)
            else:
                log.error(f'kube_event_loop: saw unexpected event_type {event_type} in {event}')

kube_thread = threading.Thread(target=run_forever, args=(kube_event_loop,))
kube_thread.start()

# debug/reloader must run in main thread
# see: https://stackoverflow.com/questions/31264826/start-a-flask-application-in-separate-thread
# flask_thread = threading.Thread(target=flask_event_loop)
# flask_thread.start()
run_forever(flask_event_loop)
