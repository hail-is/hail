import requests
from .requests_helper import raise_on_failure, filter_params


class API():
    def __init__(self, timeout=None, cookies=None, headers=None):
        """
        Python API for accessing the batch server's HTTP endpoints.

        Parameters
        ----------
        timeout : :obj:`int` or :obj:`float`
            timeout, in seconds, passed to ``requests`` calls
        """
        if timeout is None:
            timeout = 60
        self.timeout = timeout
        if cookies is None:
            cookies = {}
        self.cookies = cookies
        if headers is None:
            headers = {}
        self.headers = headers

    def http(self, verb, url, json=None, timeout=None, cookies=None, headers=None, json_response=True, params=None):
        if timeout is None:
            timeout = self.timeout
        if cookies is None:
            cookies = self.cookies
        if headers is None:
            headers = self.headers
        if json is not None:
            response = verb(url, json=json, timeout=timeout, cookies=cookies, headers=headers, params=params)
        else:
            response = verb(url, timeout=timeout, cookies=cookies, headers=headers, params=params)
        raise_on_failure(response)
        if json_response:
            return response.json()
        return response

    def post(self, *args, **kwargs):
        return self.http(requests.post, *args, **kwargs)

    def get(self, *args, **kwargs):
        return self.http(requests.get, *args, **kwargs)

    def patch(self, *args, **kwargs):
        return self.http(requests.patch, *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.http(requests.delete, *args, **kwargs)

    def create_job(self, url, spec, attributes, batch_id, callback, parent_ids,
                   scratch_folder, input_files, output_files, copy_service_account_name,
                   always_run):
        doc = {
            'spec': spec,
            'parent_ids': parent_ids,
            'always_run': always_run
        }
        if attributes:
            doc['attributes'] = attributes
        if batch_id:
            doc['batch_id'] = batch_id
        if callback:
            doc['callback'] = callback
        if scratch_folder:
            doc['scratch_folder'] = scratch_folder
        if input_files:
            doc['input_files'] = input_files
        if output_files:
            doc['output_files'] = output_files
        if copy_service_account_name:
            doc['copy_service_account_name'] = copy_service_account_name

        return self.post(f'{url}/jobs/create', json=doc)

    def list_batches(self, url, complete, success, attributes):
        params = filter_params(complete, success, attributes)
        return self.get(f'{url}/batches', params=params)

    def list_jobs(self, url, complete, success, attributes):
        params = filter_params(complete, success, attributes)
        return self.get(f'{url}/jobs', params=params)

    def get_job(self, url, job_id):
        return self.get(f'{url}/jobs/{job_id}')

    def get_job_log(self, url, job_id):
        return self.get(f'{url}/jobs/{job_id}/log')

    def delete_job(self, url, job_id):
        return self.delete(f'{url}/jobs/{job_id}/delete', json_response=False)

    def cancel_job(self, url, job_id):
        return self.patch(f'{url}/jobs/{job_id}/cancel', json_response=False)

    def create_batch(self, url, attributes, callback, ttl):
        doc = {}
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if ttl:
            doc['ttl'] = ttl
        return self.post(f'{url}/batches/create', json=doc)

    def get_batch(self, url, batch_id):
        return self.get(f'{url}/batches/{batch_id}')

    def close_batch(self, url, batch_id):
        self.patch(f'{url}/batches/{batch_id}/close', json_response=False)

    def delete_batch(self, url, batch_id):
        self.delete(f'{url}/batches/{batch_id}', json_response=False)

    def cancel_batch(self, url, batch_id):
        self.patch(f'{url}/batches/{batch_id}/cancel', json_response=False)
