import subprocess as sp


__ARG_MAX = None


def arg_max():
    global __ARG_MAX
    if __ARG_MAX is None:
        __ARG_MAX = int(sp.check_output(['getconf', 'ARG_MAX']))
    return __ARG_MAX


DEFAULT_SHELL = '/bin/bash'
