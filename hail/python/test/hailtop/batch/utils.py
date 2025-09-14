import inspect
import os

from hailtop import __pip_version__
from hailtop.batch import Batch

DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'ubuntu:24.04')
PYTHON_DILL_IMAGE = 'hailgenetics/python-dill:3.11-slim'
HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{__pip_version__}')
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
