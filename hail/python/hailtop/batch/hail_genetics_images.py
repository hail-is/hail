import sys
import os

from .exceptions import BatchException


HAIL_GENETICS = 'hailgenetics/'
HAIL_GENETICS_IMAGES = [
    HAIL_GENETICS + name
    for name in ('hail', 'genetics', 'python-dill')]


def hail_genetics_python_dill_image():
    try:
        return os.environ['HAIL_GENETICS_PYTHON_DILL_IMAGE']
    except KeyError:
        version = sys.version_info
        if version.major != 3 or version.minor not in (6, 7, 8):
            raise BatchException(
                "You must specify 'image' for Python jobs and BatchPoolExecutor if you are using "
                f"a Python version other than 3.6, 3.7, or 3.8 (you are using {version})")
        return f'hailgenetics/python-dill:{version.major}.{version.minor}-slim'
