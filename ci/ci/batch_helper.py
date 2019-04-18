import json

import requests

from .ci_logging import log
from .git_state import FQSHA


def try_to_delete_job(job):
    try:
        job.delete()
    except requests.exceptions.HTTPError as e:
        log.warning(f'could not delete job {job.id} due to {e}')


# job_ordering(x, y) > 0 if x is closer to finishing or has a larger id
def job_ordering(job1, job2):
    x = job1.cached_status()['state']
    y = job2.cached_status()['state']
    if x == 'Complete':
        if y == 'Complete':
            return job1.id - job2.id
        else:
            return 1
    elif x == 'Cancelled':
        if y == 'Cancelled':
            return job1.id - job2.id
        else:
            assert y in ('Created', 'Ready', 'Complete'), y
            return -1
    else:
        assert x in ('Created', 'Ready'), x
        if y == 'Created' or y == 'Ready':
            return job1.id - job2.id
        elif y == 'Complete':
            return -1
        else:
            assert y == 'Cancelled', y
            return 1


def short_str_build_job(job):
    state = job.cached_status()['state']
    attr = job.attributes
    assert 'target' in attr, f'{attr} {job.id}'
    assert 'source' in attr, f'{attr} {job.id}'
    assert 'type' in attr, f'{attr} {job.id}'
    assert 'image' in attr, f'{attr} {job.id}'
    target = FQSHA.from_json(json.loads(attr['target']))
    source = FQSHA.from_json(json.loads(attr['source']))
    return (
        f'[buildjob {job.id}]{state};'
        f'{target.short_str()}'
        f'..'
        f'{source.short_str()};'
        f'{attr["type"]};{attr["image"]};'
    )
