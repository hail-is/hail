import fcntl
import os
import argparse
import subprocess as sp

from pathlib import Path
from hailtop.utils import blocking_to_async


class Flock:
    def __init__(self, path, pool=None, nonblock=False):
        self.path = Path(path).resolve()
        self.lock_path = self.path.parent
        self.pool = pool
        self.flock_flags = fcntl.LOCK_EX
        if nonblock:
            self.flock_flags |= fcntl.LOCK_NB
        self.fd = -1

    def __enter__(self):
        self.lock_path.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.lock_path, os.O_RDONLY)
        fcntl.flock(self.fd, self.flock_flags)
        return self

    def __exit__(self, type, value, traceback):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)

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
    parser.add_argument('-n', dest='nonblock', action='store_true')
    args = parser.parse_args()

    with Flock(args.path):
        try:
            sp.check_output(args.command, stderr=sp.STDOUT, shell=True)
        except sp.CalledProcessError as e:
            print(e.output)
            raise e
