import json
import time
import random
import requests

def create_job(url, name, image, command=None, args=None, env=None, batch=None):
    d = {
        'name': name,
        'image': image,
    }
    if command:
        d['command'] = command
    if args:
        d['args'] = args
    if env:
        d['env'] = env
    if batch:
        d['batch'] = batch

    r = requests.post(url + '/jobs/create', json = d)
    r.raise_for_status()
    return r.json()

def get_job(url, job_id):
    r = requests.get(url + '/jobs/{}'.format(job_id))
    r.raise_for_status()
    return r.json()

def cancel_job(url, job_id):
    r = requests.post(url + '/jobs/{}/cancel'.format(job_id))
    r.raise_for_status()
    return r.json()

def get_batch(url, batch):
    r = requests.get(url + '/batches/{}'.format(batch))
    r.raise_for_status()
    return r.json()

def wait(job_id):
    i = 0
    while True:
        r = get_job(job_id)
        if r['state'] == 'Complete':
            return r
        j = random.randrange(2 ** i)
        time.sleep(0.100 * j)
        # max 5.12s
        if i < 9:
            i = i + 1

def wait_batch(batch):
    i = 0
    while True:
        r = get_batch(batch)
        if r['jobs'].get('Created', 0) == 0:
            return r
        j = random.randrange(2 ** i)
        time.sleep(0.100 * j)
        # max 5.12s
        if i < 9:
            i = i + 1
