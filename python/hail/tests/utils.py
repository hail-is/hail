import os
import sys
import hail


def startTestHailContext():
    hail.init(master='local[2]', min_block_size=0, quiet=False)


def stopTestHailContext():
    hail.stop()


_test_dir = None
_doctest_dir = None


def resource(filename):
    global _test_dir
    if _test_dir is None:
        path = '.'
        while not os.path.exists(os.path.join(path, 'LICENSE')):
            path = os.path.join(path, '..')
        _test_dir = os.path.join(path, 'src', 'test', 'resources')

    return os.path.join(_test_dir, filename)


def doctest_resource(filename):
    global _doctest_dir
    if _doctest_dir is None:
        path = '.'
        while not os.path.exists(os.path.join(path, 'LICENSE')):
            path = os.path.join(path, '..')
        _doctest_dir = os.path.join(path, 'python', 'hail', 'docs', 'data')

    return os.path.join(_doctest_dir, filename)
