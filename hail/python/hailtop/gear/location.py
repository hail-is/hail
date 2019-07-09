import os
import logging

log = logging.getLogger('gear')


def get_location():
    location = os.environ.get('HAIL_LOCATION', 'external')
    assert location in ('k8s', 'gce', 'external')
    log.info(f'location = {location}')
    return location
