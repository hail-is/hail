import sys
import os
import time
import random
import threading
from flask import Flask, request, jsonify, abort

import kubernetes as kube
import cerberus
import requests

from ..data import JobSpec

from .batch import Batch
from .dag import Dag
from .globals import max_id, pod_name_job, job_id_job, _log_path, _read_file
from .globals import batch_id_batch, next_id, REFRESH_INTERVAL_IN_SECONDS
from .globals import POD_NAMESPACE, INSTANCE_ID, dag_id_dag
from .job import Job
from .kubernetes import v1
from .log import log
from .user_error import UserError

if not os.path.exists('logs'):
    os.mkdir('logs')
else:
    if not os.path.isdir('logs'):
        raise OSError('logs exists but is not a directory')


app = Flask('batch')


@app.errorhandler(UserError)
def handle_invalid_usage(err):
    return jsonify(err.data), err.status_code


@app.route('/jobs/create', methods=['POST'])
def create_job():
    doc = request.json

    if not JobSpec.validator.validate(doc):
        abort(404, 'invalid request: {}'.format(JobSpec.validator.errors))

    job_spec = JobSpec.from_json(doc)

    if job_spec.batch_id:
        if job_spec.batch_id not in batch_id_batch:
            abort(404, 'valid request: batch_id {} not found'.format(job_spec.batch_id))

    return jsonify(Job(job_spec).to_json())


@app.route('/jobs', methods=['GET'])
def get_job_list():
    return jsonify([job.to_json() for _, job in job_id_job.items()])


@app.route('/jobs/<int:job_id>', methods=['GET'])
def get_job(job_id):
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    return jsonify(job.to_json())


@app.route('/jobs/<int:job_id>/log', methods=['GET'])
def get_job_log(job_id):  # pylint: disable=R1710
    if job_id > max_id():
        abort(404)

    job = job_id_job.get(job_id)
    if job:
        job_log = job._read_log()
        if job_log:
            return job_log
    else:
        fname = _log_path(job_id)
        if os.path.exists(fname):
            return _read_file(fname)

    abort(404)


@app.route('/jobs/<int:job_id>/delete', methods=['DELETE'])
def delete_job(job_id):
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    job.delete()
    return jsonify({})


@app.route('/jobs/<int:job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    job.cancel()
    return jsonify({})


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
    validator = cerberus.Validator(schema)
    if not validator.validate(parameters):
        abort(404, 'invalid request: {}'.format(validator.errors))

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


@app.route('/dag/create', methods=['POST'])
def create_dag():
    doc = request.json
    if not Dag.validator.validate(doc):
        abort(400, 'invalid request: {}'.format(Dag.validator.errors))
    dag = Dag.from_json(next_id(), doc)
    dag_id_dag[dag.id] = dag
    return str(dag.id), 201


@app.route('/dag/<int:dag_id>', methods=['GET'])
def get_dag(dag_id):
    dag = dag_id_dag.get(dag_id)
    if not dag:
        abort(404)
    return jsonify(dag.to_get_json())


def update_job_with_pod(job, pod):
    if pod:
        if pod.status.container_statuses:
            assert len(pod.status.container_statuses) == 1
            container_status = pod.status.container_statuses[0]
            assert container_status.name == 'default'

            if container_status.state and container_status.state.terminated:
                job.mark_complete(pod)
    else:
        job.mark_unscheduled()


@app.route('/pod_changed', methods=['POST'])
def pod_changed():
    parameters = request.json

    pod_name = parameters['pod_name']

    job = pod_name_job.get(pod_name)
    if job and not job.is_complete():
        try:
            pod = v1.read_namespaced_pod(
                pod_name,
                POD_NAMESPACE,
                _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
        except kube.client.rest.ApiException as exc:
            if exc.status == 404:
                pod = None
            else:
                raise

        update_job_with_pod(job, pod)

    return '', 204


@app.route('/refresh_k8s_state', methods=['POST'])
def refresh_k8s_state():
    log.info('started k8s state refresh')

    pods = v1.list_namespaced_pod(
        POD_NAMESPACE,
        label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}',
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)

    seen_pods = set()
    for pod in pods.items:
        pod_name = pod.metadata.name
        seen_pods.add(pod_name)

        job = pod_name_job.get(pod_name)
        if job and not job.is_complete():
            update_job_with_pod(job, pod)

    for pod_name, job in pod_name_job.items():
        if pod_name not in seen_pods:
            update_job_with_pod(job, None)

    log.info('k8s state refresh complete')

    return '', 204


def run_forever(target, *args, **kwargs):
    # target should be a function
    target_name = target.__name__

    expected_retry_interval_ms = 15 * 1000
    while True:
        start = time.time()
        try:
            log.info(f'run_forever: run target {target_name}')
            target(*args, **kwargs)
            log.info(f'run_forever: target {target_name} returned')
        except Exception:  # pylint: disable=W0703
            log.error(f'run_forever: target {target_name} threw exception', exc_info=sys.exc_info())
        end = time.time()

        run_time_ms = int((end - start) * 1000 + 0.5)
        sleep_duration_ms = random.randrange(expected_retry_interval_ms * 2) - run_time_ms
        if sleep_duration_ms > 0:
            log.debug(f'run_forever: {target_name}: sleep {sleep_duration_ms}ms')
            time.sleep(sleep_duration_ms / 1000.0)


def flask_event_loop():
    app.run(threaded=False, host='0.0.0.0')


def kube_event_loop():
    watch = kube.watch.Watch()
    stream = watch.stream(
        v1.list_namespaced_pod,
        POD_NAMESPACE,
        label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
    for event in stream:
        pod = event['object']
        name = pod.metadata.name
        requests.post('http://127.0.0.1:5000/pod_changed', json={'pod_name': name}, timeout=120)


def polling_event_loop():
    time.sleep(1)
    while True:
        try:
            response = requests.post('http://127.0.0.1:5000/refresh_k8s_state', timeout=120)
            response.raise_for_status()
        except requests.HTTPError as exc:
            log.error(f'Could not poll due to exception: {exc}, text: {exc.response.text}')
        except Exception as exc:  # pylint: disable=W0703
            log.error(f'Could not poll due to exception: {exc}')
        time.sleep(REFRESH_INTERVAL_IN_SECONDS)


def serve():
    kube_thread = threading.Thread(target=run_forever, args=(kube_event_loop,))
    kube_thread.start()

    polling_thread = threading.Thread(target=run_forever, args=(polling_event_loop,))
    polling_thread.start()

    # debug/reloader must run in main thread
    # see: https://stackoverflow.com/questions/31264826/start-a-flask-application-in-separate-thread
    # flask_thread = threading.Thread(target=flask_event_loop)
    # flask_thread.start()
    run_forever(flask_event_loop)

    kube_thread.join()
