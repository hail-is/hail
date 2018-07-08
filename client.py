import json
import time
import random
import requests

_url = 'http://batch'
def configure(url):
    global _url
    
    _url = url

def create_job(name, image, command=None, args=None, env=None):
    d = {
        'name': name,
        'image': image,
    }
    if command:
        d['command'] = command
    if args:
        d['command'] = args
    if env:
        d['command'] = env

    r = requests.post(_url + '/jobs/create', json = d)
    r.raise_for_status()
    return r.json()

def get_job(job_id):
    r = requests.get(_url + '/jobs/{}'.format(job_id))
    r.raise_for_status()
    return r.json()

def cancel(job_id):
    r = requests.post(_url + '/jobs/{}/cancel'.format(job_id))
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
