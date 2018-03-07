import os
import sys
import hail


def defaultSetUpModule():
    hail.init(master='local[2]', min_block_size=0, quiet=True)


def defaultTearDownModule():
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
        sys.stderr.write('Test dir relative path is {}'.format(_test_dir))

    return os.path.join(_test_dir, filename)


def doctest_resource(filename):
    global _doctest_dir
    if _doctest_dir is None:
        path = '.'
        while not os.path.exists(os.path.join(path, 'LICENSE')):
            path = os.path.join(path, '..')
        _doctest_dir = os.path.join(path, 'python', 'hail', 'docs', 'data')
        sys.stderr.write('Doctest dir relative path is {}'.format(_doctest_dir))

    return os.path.join(_doctest_dir, filename)
