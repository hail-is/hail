import json
import time
import random
import requests

def create_job(url, spec, attributes, batch_id, callback):
    d = {'spec': spec}
    if attributes:
        d['attributes'] = attributes
    if batch_id:
        d['batch_id'] = batch_id
    if callback:
        d['callback'] = callback

    r = requests.post(url + '/jobs/create', json = d)
    r.raise_for_status()
    return r.json()

def list_jobs(url):
    r = requests.get(url + '/jobs')
    r.raise_for_status()
    return r.json()

def get_job(url, job_id):
    r = requests.get(url + '/jobs/{}'.format(job_id))
    r.raise_for_status()
    return r.json()

def delete_job(url, job_id):
    r = requests.delete(url + '/jobs/{}/delete'.format(job_id))
    r.raise_for_status()
    return r.json()

def cancel_job(url, job_id):
    r = requests.post(url + '/jobs/{}/cancel'.format(job_id))
    r.raise_for_status()
    return r.json()

def create_batch(url, attributes):
    d = {}
    if attributes:
        d['attributes'] = attributes
    r = requests.post(url + '/batches/create', json = d)
    r.raise_for_status()
    return r.json()

def get_batch(url, batch_id):
    r = requests.get(url + '/batches/{}'.format(batch_id))
    r.raise_for_status()
    return r.json()

def delete_batch(url, batch_id):
    r = requests.delete(url + '/batches/{}'.format(batch_id))
    r.raise_for_status()
    return r.json()
