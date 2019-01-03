import abc
import os
from .resource import Resource, ResourceGroup
from .utils import get_sha


class Backend(object):
    @abc.abstractmethod
    def tmp_dir(self):
        return

    @abc.abstractmethod
    def run(self, pipeline):
        return

    @abc.abstractmethod
    def cp(self, src, dest):
        return

    @abc.abstractmethod
    def mv(self, src, dest):
        return


class LocalBackend(Backend):
    def __init__(self, tmp_dir='/tmp/', delete_on_exit=True):
        self._tmp_dir = tmp_dir
        self._delete_on_exit = delete_on_exit

    def run(self, pipeline):
        from .pipeline import Pipeline

        script = ['#! /usr/bash',
                  'set -ex',
                  '\n',
                  '# define tmp directory',
                  f"{Pipeline._tmp_dir_varname}={self.tmp_dir()}",
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

        if self._delete_on_exit:
            script += ['# remove tmp directory',
                       f'rm -r ${{{Pipeline._tmp_dir_varname}}}']

        print("\n".join(script)) # FIXME: replace with subprocess.call()

    def tmp_dir(self):
        def _get_random_name():
            directory = self._tmp_dir + '/pipeline.{}/'.format(get_sha(8))

            if os.path.isdir(directory):
                _get_random_name()
            else:
                os.mkdir(directory)
                return directory

        return _get_random_name()

    def cp(self, src, dest): # FIXME: symbolic links? support gsutil?
        return f"cp {src} {dest}"

    def mv(self, src, dest):
        return f"mv {src} {dest}"

