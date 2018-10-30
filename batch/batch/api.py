import json
import time
import random
import requests

class API():
    def __init__(self, timeout=60):
        """
        Python API for accessing the batch server's HTTP endpoints.

        Parameters
        ----------
        timeout : :obj:`int` or :obj:`float`
            timeout, in seconds, passed to ``requests`` calls
        """
        self.timeout = timeout

    def create_job(url, spec, attributes, batch_id, callback):
        d = {'spec': spec}
        if attributes:
            d['attributes'] = attributes
        if batch_id:
            d['batch_id'] = batch_id
        if callback:
            d['callback'] = callback

        r = requests.post(url + '/jobs/create', json=d, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_jobs(url):
        r = requests.get(url + '/jobs', timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_job(url, job_id):
        r = requests.get(url + '/jobs/{}'.format(job_id), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_job_log(url, job_id):
        r = requests.get(url + '/jobs/{}/log'.format(job_id), timeout=self.timeout)
        r.raise_for_status()
        return r.text

    def delete_job(url, job_id):
        r = requests.delete(url + '/jobs/{}/delete'.format(job_id), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def cancel_job(url, job_id):
        r = requests.post(url + '/jobs/{}/cancel'.format(job_id), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create_batch(url, attributes):
        d = {}
        if attributes:
            d['attributes'] = attributes
        r = requests.post(url + '/batches/create', json=d, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_batch(url, batch_id):
        r = requests.get(url + '/batches/{}'.format(batch_id), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def delete_batch(url, batch_id):
        r = requests.delete(url + '/batches/{}'.format(batch_id), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def refresh_k8s_state(url):
        r = requests.post(url + '/refresh_k8s_state', timeout=self.timeout)
        r.raise_for_status()


__default_api = API()


def create_job(url, spec, attributes, batch_id, callback):
    return __default_api.create_job(url, spec, attributes, batch_id, callback)


def list_jobs(url):
    return __default_api.list_jobs(url)


def get_job(url, job_id):
    return __default_api.get_job(url, job_id)


def get_job_log(url, job_id):
    return __default_api.get_job_log(url, job_id)


def delete_job(url, job_id):
    return __default_api.delete_job(url, job_id)


def cancel_job(url, job_id):
    return __default_api.cancel_job(url, job_id)


def create_batch(url, attributes):
    return __default_api.create_batch(url, attributes)


def get_batch(url, batch_id):
    return __default_api.get_batch(url, batch_id)


def delete_batch(url, batch_id):
    return __default_api.delete_batch(url, batch_id)


def refresh_k8s_state(url):
    return __default_api.refresh_k8s_state(url)
