import os
import logging
import threading
from flask import Flask, request, jsonify, abort, url_for
import kubernetes as kube
import cerberus

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

        created_pod = v1.create_namespaced_pod('default', self.pod)
        self._pod_name = created_pod.metadata.name
        pod_name_job[self._pod_name] = self

        log.info('created pod name: {} for job {}'.format(self._pod_name, self.id))

    def _cancel_pod(self):
        if self._pod_name:
            try:
                v1.delete_namespaced_pod(self._pod_name, 'default', kube.client.V1DeleteOptions())
            except kube.client.rest.ApiException as e:
                if e.status == 404 and e.reason = 'NotFound':
                    pass
                else:
                    raise
            del pod_name_job[self._pod_name]
            self._pod_name = None

    def __init__(self, parameters):
        self.name = parameters['name']
        
        self.id = next_id()
        job_id_job[self.id] = self
        log.info('created job {}'.format(self.id))

        self._state = 'Created'

        image = parameters['image']
        command = parameters.get('command')
        args = parameters.get('args')
        env = parameters.get('env')
        if env:
            env = [kube.client.V1EnvVar(name = k, value = v) for (k, v) in env.items()]
        self.pod = kube.client.V1Pod(
            metadata = kube.client.V1ObjectMeta(generate_name = self.name + '-'),
            spec = kube.client.V1PodSpec(
                containers = [
                    kube.client.V1Container(
                        name = 'default',
                        image = image,
                        command = command,
                        args = args,
                        env = env)
                ],
                restart_policy = 'Never'))
        self._pod_name = None
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
            self.name, self.exit_code, self.log))
        self.set_state('Complete')

    def to_json(self):
        result = {
            'id': self.id,
            'state': self._state,
        }
        if self._state == 'Complete':
            result['exit_code'] = self.exit_code
            result['log'] = self.log
        return result

app = Flask('batch')

@app.route('/jobs/create', methods=['POST'])
def schedule():
    parameters = request.json

    schema = {
        'name': {'type': 'string', 'required': True},
        'image': {'type': 'string', 'required': True},
        'command': {'type': 'list', 'schema': {'type': 'string'}},
        'args': {'type': 'list', 'schema': {'type': 'string'}},
        'env': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        }
    }
    v = cerberus.Validator(schema)
    if (not v.validate(parameters)):
        abort(404, 'invalid request: {}'.format(v.errors))

    job = Job(parameters)
    result = {'id': job.id, 'location': url_for('get_job', job_id=job.id, _external=True)}
    return jsonify(result)

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

def flask_event_loop():
    app.run(debug=True, host='0.0.0.0')

def kube_event_loop():
    stream = kube.watch.Watch().stream(v1.list_namespaced_pod, 'default')
    for event in stream:
        # print(event)
        event_type = event['type']

        pod = event['object']
        name = pod.metadata.name

        job = pod_name_job.get(name)
        if job and not job.is_complete():
            if event_type == 'DELETE':
                job.mark_unscheduled()
            else:
                assert event_type == 'ADDED' or event_type == 'MODIFIED'
                if pod.status.container_statuses:
                    assert len(pod.status.container_statuses) == 1
                    container_status = pod.status.container_statuses[0]
                    assert container_status.name == 'default'

                    if container_status.state and container_status.state.terminated:
                        job.mark_complete(pod)

kube_thread = threading.Thread(target=kube_event_loop)
kube_thread.start()

# debug/reloader must run in main thread
# see: https://stackoverflow.com/questions/31264826/start-a-flask-application-in-separate-thread
# flask_thread = threading.Thread(target=flask_event_loop)
# flask_thread.start()
flask_event_loop()
