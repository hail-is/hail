import subprocess as sp

__ARG_MAX = None


def arg_max():
    global __ARG_MAX
    if __ARG_MAX is None:
        __ARG_MAX = int(sp.check_output(['getconf', 'ARG_MAX']))
    return __ARG_MAX


class BatchException(Exception):
    def __init__(self, msg=''):
        self.msg = msg
        super(BatchException, self).__init__(msg)
