from __future__ import print_function  # Python 2 and 3 print compatibility

import datetime
import inspect
import itertools
from collections import OrderedDict
from decorator import decorator
from hail.utils import hadoop_write, wrap_to_list
from hail.java import Env


def parse_args(f, args, kwargs):
    argspec = inspect.getargspec(f)
    kwargs_not_default = OrderedDict({})

    if argspec.defaults:
        n_postnl_args = len(argspec.args) - len(argspec.defaults)
        kwargs_not_default.update(OrderedDict({k:v for k, (v, d) in zip(argspec.args[n_postnl_args:], zip(args[n_postnl_args - 1:], argspec.defaults)) if v != d}))
        postnl_args = args[:n_postnl_args - 1]
    else:
        postnl_args = args

    kwargs_not_default.update(OrderedDict({k:v for k,v in kwargs.iteritems()}))

    return postnl_args, kwargs_not_default


@decorator
def record_init(func, obj, *args, **kwargs):
    postnl_args, kwargs_not_default = parse_args(func, args, kwargs)
    h = History.from_init(type(obj).__name__, postnl_args, kwargs_not_default)
    func(obj, *args, **kwargs)
    obj._set_history(h)


@decorator
def record_method(func, obj, *args, **kwargs):
    postnl_args, kwargs_not_default = parse_args(func, args, kwargs)

    def set_history(item, index=None, key_name=None):
        if isinstance(item, HistoryMixin):
            item._set_history(obj
                              ._get_history()
                              .add_method(func.__name__, postnl_args, kwargs_not_default, index=index, key_name=key_name))

    result = func(obj, *args, **kwargs)
    if isinstance(result, dict):
        for k, r in result.iteritems():
            set_history(r, key_name=k)
    elif isinstance(result, list) or isinstance(result, tuple):
        for i, r in enumerate(result):
            set_history(r, index=i)
    else:
        set_history(result)
    return result


@decorator
def record_classmethod(func, cls, *args, **kwargs):
    postnl_args, kwargs_not_default = parse_args(func, args, kwargs)
    result = func(cls, *args, **kwargs)
    result._set_history(History.from_classmethod(cls.__name__, func.__name__, postnl_args, kwargs_not_default))
    return result


def write_history(path_arg_name, is_dir=False):
    def _write(f, obj, *args, **kwargs):
        postnl_args, kwargs_not_default = parse_args(f, args, kwargs)
        argnames = inspect.getcallargs(f, obj, *args, **kwargs)
        result = f(obj, *args, **kwargs)

        output_path = argnames[path_arg_name]
        if is_dir:
            output_path = output_path + "/history.txt"
        else:
            output_path += ".history.txt"

        (obj._history
         .add_method(f.__name__, postnl_args, kwargs_not_default)
         .write(output_path))

        return result
    return decorator(_write)


def format_args(arg, stmts):
    if isinstance(arg, list):
        return [format_args(a, stmts) for a in arg]
    elif isinstance(arg, tuple):
        return tuple([format_args(a, stmts) for a in arg])
    elif isinstance(arg, dict):
        return {format_args(k, stmts): format_args(v, stmts) for k, v in arg.iteritems()}
    else:
        if isinstance(arg, HistoryMixin):
            h = arg._get_history()
            stmts += remove_dup_stmts(stmts, h.stmts)
            return h
        else:
            return arg


def remove_dup_stmts(stmts1, stmts2):
    return [s2 for s1, s2 in itertools.izip_longest(stmts1, stmts2) if s2 and s1 != s2]


class History(object):
    def __init__(self, expr="", stmts=[]):
        self.expr = expr
        self.stmts = stmts

    def set_varid(self, id):
        return History(id, self.stmts + ['{} = ({})'.format(id, self.expr)])

    def add_method(self, f_name, args, kwargs, index=None, key_name=None):
        stmts = list(self.stmts) # make a copy
        f_args = [repr(format_args(a, stmts)) for a in args] + ["{}={}".format(k, repr(format_args(v, stmts))) for k, v in kwargs.iteritems()]
        expr = self.expr + ("\n    ." + f_name + "(" + ", ".join(f_args) + ")")
        if index:
            expr += "[{}]".format(index)
        elif key_name:
            expr += "[{}]".format(repr(key_name))
        return History(expr, stmts)

    @staticmethod
    def from_classmethod(cls_name, f_name, args, kwargs):
        stmts = []
        f_args = [repr(format_args(a, stmts)) for a in args] + ["{}={}".format(k, repr(format_args(v, stmts))) for k, v in kwargs.iteritems()]
        expr = cls_name + "." + f_name + "(" + ", ".join(f_args) + ")"
        return History(expr, stmts)

    @staticmethod
    def from_init(cls_name, args, kwargs):
        stmts = []
        f_args = [repr(format_args(a, stmts)) for a in args] + ["{}={}".format(k, repr(format_args(v, stmts))) for k, v in kwargs.iteritems()]
        expr = cls_name + "(" + ", ".join(f_args) + ")"
        return History(expr, stmts)

    def write(self, file):
        with hadoop_write(file) as f:
            f.write(str(self) + "\n")

    def __repr__(self):
        return self.expr

    def __str__(self):
        now = datetime.datetime.now()
        history = "# {}\n# version: {}\n\n".format(now.isoformat(), Env._hc.version)
        for stmt in self.stmts:
            history += (stmt + "\n\n")
        history += ("(" + self.expr + ")")
        try:
            import autopep8
            return autopep8.fix_code(history)
        except:
            return history


class HistoryMixin(object):
    def __init__(self):
        self._history = None

    def _set_history(self, history):
        self._history = history

    def _get_history(self):
        return self._history

    def with_id(self, id):
        self._set_history(self._history.set_varid(id))
        return self