import os
import inspect
from hailtop import pip_version
from hailtop.batch import Batch


DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'ubuntu:22.04')
PYTHON_DILL_IMAGE = 'hailgenetics/python-dill:3.9-slim'
HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}')
REQUESTER_PAYS_PROJECT = os.environ.get('GCS_REQUESTER_PAYS_PROJECT')


def batch(backend, **kwargs):
    name_of_test_method = inspect.stack()[1][3]
    return Batch(
        name=name_of_test_method,
        backend=backend,
        default_image=DOCKER_ROOT_IMAGE,
        attributes={'foo': 'a', 'bar': 'b'},
        **kwargs,
    )
