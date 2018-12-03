import abc
import random, string
import os

def get_sha(k):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))


class Backend(object):
    @abc.abstractmethod
    def run(self, pipeline):
        return

    @abc.abstractmethod
    def cp_cmd_template(self):
        return

    @abc.abstractmethod
    def mv_cmd_template(self):
        return


class LocalBackend(Backend):
    def __init__(self, tmp_dir='/tmp/'):
        self._tmp_dir = tmp_dir

    def run(self, pipeline):
        script = ['#! /usr/bash',
                  'set -ex',
                  '\n']

        def define_resource(r):
            assert resource._value is not None
            if isinstance(r._value, list):
                init = "{}=({})".format(r._uid,
                                        " ".join([str(x._value) for x in r._value]))
            else:
                init = f"{r._uid}={r._value}"
            return init

        def reference_resource(r):
            if isinstance(r._value, list):
                return f"${{{r._uid}[*]}}"
            else:
                return f"${r._uid}"

        for task in pipeline._tasks:
            script.append(f"# {task._uid} {task._label if task._label else ''}")
            for _, resource in task._resources.items():
                script.append(define_resource(resource))
            script += task._render_command(reference_resource) + ["\n"]

        # replace with subprocess.call()
        print("\n".join(script))

    def temp_file(self, prefix=None, suffix=None):
        def _get_random_name():
            file = self._tmp_dir + '{}{}{}'.format(prefix if prefix else '',
                                                   get_sha(6),
                                                   suffix if suffix else '')
            if os.path.exists(file):
                _get_random_name()
            else:
                return file

        return _get_random_name()

    def cp_cmd_template(self):
        return "cp {{src}} {{dest}}"

    def mv_cmd_template(self):
        return "mv {{src}} {{dest}}"

