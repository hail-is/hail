import fcntl
import os
import argparse
import subprocess as sp

from hailtop.utils import blocking_to_async


class Flock:
    def __init__(self, path, pool=None):
        self.path = os.path.abspath(path)
        self.pool = pool

        self.flock_flags = fcntl.LOCK_EX
        self.fds = []

        if os.path.isdir(self.path):
            self.path = self.path.rstrip('/') + '/'

    def __enter__(self):
        dirname, filename = os.path.split(self.path)

        components = dirname.split('/')
        for i in range(2, len(components) + 1):
            path = '/'.join(components[:i])
            os.makedirs(path, exist_ok=True)
            fd = os.open(path, os.O_RDONLY)
            self.fds.append(fd)
            fcntl.flock(fd, self.flock_flags)

        if filename:
            fd = os.open(self.path, os.O_CREAT)
            self.fds.append(fd)
            fcntl.flock(fd, self.flock_flags)

        return self

    def __exit__(self, type, value, traceback):
        for fd in reversed(self.fds):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    async def __aenter__(self):
        assert self.pool
        return await blocking_to_async(self.pool, self.__enter__)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        assert self.pool
        return await blocking_to_async(self.pool, self.__exit__, exc_type, exc_val, exc_tb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-c', dest='command', type=str, required=True)
    args = parser.parse_args()

    with Flock(args.path):
        try:
            sp.check_output(args.command, stderr=sp.STDOUT, shell=True)
        except sp.CalledProcessError as e:
            print(e.output)
            raise e
