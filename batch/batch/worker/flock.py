import fcntl
import os
import argparse
import subprocess as sp


class Flock:
    def __init__(self, path, nonblock=False):
        self.path = os.path.abspath(path)
        self.flock_flags = fcntl.LOCK_EX
        self.fds = []

        if nonblock:
            self.flock_flags |= fcntl.LOCK_NB

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-c', dest='command', type=str, required=True)
    parser.add_argument('-n', dest='nonblock', action='store_true')
    args = parser.parse_args()

    with Flock(args.path, args.nonblock):
        try:
            sp.check_output(args.command, stderr=sp.STDOUT, shell=True)
        except sp.CalledProcessError as e:
            print(e.output)
            raise e
