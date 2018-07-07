import logging
import threading
from flask import Flask, request, jsonify
import kubernetes as kube

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('batch')

kube.config.load_incluster_config()
# kube.config.load_kube_config()
v1 = kube.client.CoreV1Api()

pod_job = {}

counter = 0
def next_uid():
    global counter
    
    counter = counter + 1
    return counter

class Job(object):
    def _create_pod(self):
        pod = kube.client.V1Pod(
            metadata = kube.client.V1ObjectMeta(generate_name = self.name + '-'),
            spec = kube.client.V1PodSpec(
                containers = [
                    kube.client.V1Container(
                        name = 'default',
                        image = self.image)
                ],
                restart_policy = 'Never'))
        pod = v1.create_namespaced_pod('default', pod)
        pod_name = pod.metadata.name
        pod_uid = pod.metadata.uid
        log.info('created pod name: {}, uid: {} for job {}'.format(pod_name, pod_uid, self.uid))
        pod_job[pod_uid] = self

    def __init__(self, name, image, callback=None):
        self.name = name
        self.uid = next_uid()
        self.image = image
        self.callback = callback

        self.state = 'Created'
        log.info('created job {}'.format(self.uid))

        self._create_pod()

    def set_state(self, new_state):
        if self.state != new_state:
            log.info('job {} changed state: {} -> {}'.format(
                self.uid,
                self.state,
                new_state))
            self.state = new_state

    def mark_unscheduled(self):
        self._create_pod()

    def mark_complete(self, exit_code):
        self.exit_code = exit_code
        log.info('job {} complete, exit_code {}'.format(self.name, exit_code))
        self.state = 'Complete'
        # FIXME call callback

app = Flask('batch')

@app.route('/schedule', methods=['POST'])
def schedule():
    parameters = request.json
    name = parameters['name']
    image = parameters['image']
    job = Job(name, image)
    result = {'uid': job.uid, 'name': job.name}
    return jsonify(result)

def flask_event_loop():
    app.run(debug=True, host='0.0.0.0')

def kube_event_loop():
    stream = kube.watch.Watch().stream(v1.list_namespaced_pod, 'default')
    for event in stream:
        # print(event)
        event_type = event['type']

        pod = event['object']
        name = pod.metadata.name
        uid = pod.metadata.uid

        job = pod_job.get(uid)
        if job and job.state != 'Complete':
            if event_type == 'DELETE':
                del pod_job[uid]
                job.mark_unscheduled()
            else:
                assert event_type == 'ADDED' or event_type == 'MODIFIED'
                if pod.status.container_statuses:
                    assert len(pod.status.container_statuses) == 1
                    container_status = pod.status.container_statuses[0]
                    assert container_status.name == 'default'

                    if container_status.state and container_status.state.terminated:
                        exit_code = container_status.state.terminated.exit_code
                        job.mark_complete(exit_code)

kube_thread = threading.Thread(target=kube_event_loop)
kube_thread.start()

# debug/reloader must run in main thread
# see: https://stackoverflow.com/questions/31264826/start-a-flask-application-in-separate-thread
# flask_thread = threading.Thread(target=flask_event_loop)
# flask_thread.start()
flask_event_loop()
