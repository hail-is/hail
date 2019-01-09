import abc
import os
import subprocess as sp

from .resource import Resource, ResourceGroup
from .utils import get_sha


class Backend(object):
    @abc.abstractmethod
    def tmp_dir(self):
        return

    @abc.abstractmethod
    def run(self, pipeline, dry_run, verbose, bg, delete_on_exit):
        return

    @abc.abstractmethod
    def cp(self, src, dest):
        return


class LocalBackend(Backend):
    def __init__(self, tmp_dir='/tmp/'):
        self._tmp_dir = tmp_dir

    def run(self, pipeline, dry_run, verbose, bg, delete_on_exit):
        wd = self.tmp_dir()

        script = ['#!/bin/bash',
                  'set -e' + 'x' if verbose else '',
                  '\n',
                  '# change cd to tmp directory',
                  f"cd {wd}",
                  '\n']

        def define_resource(r):
            if isinstance(r, str):
                r = pipeline._resource_map[r]

            if isinstance(r, Resource):
                assert r._value is not None
                init = f"{r._uid}={r._value}"
            else:
                assert isinstance(r, ResourceGroup)
                init = f"{r._uid}={r._root}"
            return init

        for task in pipeline._tasks:
            script.append(f"# {task._uid} {task._label if task._label else ''}")
            script += [define_resource(r) for _, r in task._resources.items()]
            script += task._command + ["\n"]

        script = "\n".join(script)

        if dry_run:
            print(script)

        if not dry_run:
            try:
                sp.check_output(script, shell=True) # FIXME: implement non-blocking (bg = True)
            except sp.CalledProcessError as e:
                print(e.output)
                raise e
            finally:
                if delete_on_exit:
                    sp.run(f'rm -r {wd}', shell=True)

    def tmp_dir(self):
        def _get_random_name():
            directory = self._tmp_dir + '/pipeline.{}/'.format(get_sha(8))

            if os.path.isdir(directory):
                _get_random_name()
            else:
                os.makedirs(directory, exist_ok=True)
                return directory

        return _get_random_name()

    def cp(self, src, dest): # FIXME: symbolic links? support gsutil?
        return f"cp {src} {dest}"

