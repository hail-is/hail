import requests
from .requests_helper import raise_on_failure


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

    def create_job(self, url, spec, attributes, batch_id, callback, parent_ids, scratch_bucket):
        doc = {
            'spec': spec,
            'parent_ids': parent_ids
        }
        if attributes:
            doc['attributes'] = attributes
        if batch_id:
            doc['batch_id'] = batch_id
        if callback:
            doc['callback'] = callback
        if scratch_bucket:
            doc['scratch_bucket'] = scratch_bucket

        response = requests.post(url + '/jobs/create', json=doc, timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def list_jobs(self, url):
        response = requests.get(url + '/jobs', timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def get_job(self, url, job_id):
        response = requests.get(url + '/jobs/{}'.format(job_id), timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def get_job_log(self, url, job_id):
        response = requests.get(url + '/jobs/{}/log'.format(job_id), timeout=self.timeout)
        raise_on_failure(response)
        return response.text

    def delete_job(self, url, job_id):
        response = requests.delete(url + '/jobs/{}/delete'.format(job_id), timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def cancel_job(self, url, job_id):
        response = requests.post(url + '/jobs/{}/cancel'.format(job_id), timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def create_batch(self, url, attributes, callback, ttl):
        doc = {}
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if ttl:
            doc['ttl'] = ttl
        response = requests.post(url + '/batches/create', json=doc, timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def get_batch(self, url, batch_id):
        response = requests.get(url + '/batches/{}'.format(batch_id), timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def close_batch(self, url, batch_id):
        response = requests.post(url + '/batches/{}/close'.format(batch_id), timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def delete_batch(self, url, batch_id):
        response = requests.delete(url + '/batches/{}'.format(batch_id), timeout=self.timeout)
        raise_on_failure(response)
        return response.json()

    def refresh_k8s_state(self, url):
        response = requests.post(url + '/refresh_k8s_state', timeout=self.timeout)
        raise_on_failure(response)


DEFAULT_API = API()


def create_job(url, spec, attributes, batch_id, callback, parent_ids, scratch_bucket):
    return DEFAULT_API.create_job(url, spec, attributes, batch_id, callback, parent_ids, scratch_bucket)


def list_jobs(url):
    return DEFAULT_API.list_jobs(url)


def get_job(url, job_id):
    return DEFAULT_API.get_job(url, job_id)


def get_job_log(url, job_id):
    return DEFAULT_API.get_job_log(url, job_id)


def delete_job(url, job_id):
    return DEFAULT_API.delete_job(url, job_id)


def cancel_job(url, job_id):
    return DEFAULT_API.cancel_job(url, job_id)


def create_batch(url, attributes, callback, ttl):
    return DEFAULT_API.create_batch(url, attributes, callback, ttl)


def get_batch(url, batch_id):
    return DEFAULT_API.get_batch(url, batch_id)


def close_batch(url, batch_id):
    return DEFAULT_API.close_batch(url, batch_id)


def delete_batch(url, batch_id):
    return DEFAULT_API.delete_batch(url, batch_id)


def refresh_k8s_state(url):
    return DEFAULT_API.refresh_k8s_state(url)
