from enum import Enum
import subprocess as sp
from typing import Union, List


__ARG_MAX = None


def arg_max():
    global __ARG_MAX
    if __ARG_MAX is None:
        __ARG_MAX = int(sp.check_output(['getconf', 'ARG_MAX']))
    return __ARG_MAX


DEFAULT_SHELL = '/bin/bash'


class _ANY_REGION(Enum):
    """
    Object that signifies that a job can run in any supported region
    """
    ANY_REGION = 1


ANY_REGION = _ANY_REGION.ANY_REGION

REGION_SPECIFICATION = Union[_ANY_REGION, List[str]]
