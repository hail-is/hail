from __future__ import print_function  # Python 2 and 3 print compatibility

import datetime
import inspect
import itertools
from collections import OrderedDict
from decorator import decorator
from hail.utils import hadoop_write
from hail.java import Env


def parse_args(f, args, kwargs, is_method=True):
    argspec = inspect.getargspec(f)

    arg_names = argspec.args[1:] if is_method else argspec.args
    defaults = list(argspec.defaults) if argspec.defaults else []
    n_postnl_args = len(arg_names) - len(defaults)
    defaults = n_postnl_args * [None] + defaults

    parsed_args = OrderedDict({k: v for k, (v, d) in zip(arg_names, zip(args, defaults)) if v != d})
    parsed_args.update(OrderedDict({k:v for k,v in kwargs.iteritems()}))
    return parsed_args


@decorator
def record_init(func, obj, *args, **kwargs):
    parsed_args = parse_args(func, args, kwargs)
    h = History.from_init(type(obj).__name__, parsed_args)
    func(obj, *args, **kwargs)
    obj._history = h


@decorator
def record_method(func, obj, *args, **kwargs):
    parsed_args = parse_args(func, args, kwargs)

    def set_history(item, index=None, key_name=None):
        if isinstance(item, HistoryMixin):
            item._history = (obj._history.add_method(func.__name__, parsed_args, index=index, key_name=key_name))

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
    parsed_args = parse_args(func, args, kwargs)
    result = func(cls, *args, **kwargs)
    result._history = History.from_classmethod(cls.__name__, func.__name__, parsed_args)
    return result


def write_history(path_arg_name, is_dir=False):
    def _write(f, obj, *args, **kwargs):
        parsed_args = parse_args(f, args, kwargs)
        argnames = inspect.getcallargs(f, obj, *args, **kwargs)
        result = f(obj, *args, **kwargs)

        output_path = argnames[path_arg_name]
        if is_dir:
            output_path = output_path + "/history.txt"
        else:
            output_path += ".history.txt"

        (obj._history
         .add_method(f.__name__, parsed_args)
         .write(output_path))

        return result
    return decorator(_write)


def format_args(arg, statements):
    if isinstance(arg, list):
        return [format_args(a, statements) for a in arg]
    elif isinstance(arg, tuple):
        return tuple([format_args(a, statements) for a in arg])
    elif isinstance(arg, dict):
        return {format_args(k, statements): format_args(v, statements) for k, v in arg.iteritems()}
    else:
        if isinstance(arg, HistoryMixin):
            h = arg._history
            statements += remove_dup_statements(statements, h.statements)
            return h
        else:
            return arg


def remove_dup_statements(statements1, statements2):
    return [s2 for s1, s2 in itertools.izip_longest(statements1, statements2) if s2 and s1 != s2]


class History(object):
    def __init__(self, expr="", statements=[]):
        self.expr = expr
        self.statements = statements

    def set_varid(self, id):
        return History(id, self.statements + ['{} = ({})'.format(id, self.expr)])

    def add_method(self, f_name, kwargs, index=None, key_name=None):
        statements = self.statements[:]
        f_args = ["{}={}".format(k, repr(format_args(v, statements))) for k, v in kwargs.iteritems()]
        expr = "{expr}\n.{f_name}({f_args})".format(expr=self.expr, f_name=f_name, f_args=", ".join(f_args))
        if index:
            expr += "[{}]".format(index)
        elif key_name:
            expr += "[{}]".format(repr(key_name))
        return History(expr, statements)

    @staticmethod
    def from_classmethod(cls_name, f_name, kwargs):
        statements = []
        f_args = ["{}={}".format(k, repr(format_args(v, statements))) for k, v in kwargs.iteritems()]
        expr = "{cls_name}.{f_name}({args})".format(cls_name=cls_name, f_name=f_name, args=", ".join(f_args))
        return History(expr, statements)

    @staticmethod
    def from_init(cls_name, kwargs):
        statements = []
        f_args = ["{}={}".format(k, repr(format_args(v, statements))) for k, v in kwargs.iteritems()]
        expr = "{cls_name}({args})".format(cls_name=cls_name, args=", ".join(f_args))
        return History(expr, statements)

    def write(self, file):
        with hadoop_write(file) as f:
            f.write(self.formatted() + "\n")

    def formatted(self):
        now = datetime.datetime.now()
        history = "# {}\n# version: {}\n\n".format(now.isoformat(), Env._hc.version)
        history += "from hail import *\n\n"
        for stmt in self.statements:
            history += (stmt + "\n\n")
        history += ("(" + self.expr + ")")
        try:
            import autopep8
            return autopep8.fix_code(history)
        except ImportError:
            return history

    def __repr__(self):
        return self.expr


class HistoryMixin(object):
    def __init__(self):
        self._history = None

    def with_id(self, id):
        """Set identifier for this object in the history file.

        **Examples**

        Given the following code that writes a VDS:

        .. code-block: python

            vds1 = hc.import_vcf('src/test/resources/sample.vcf').with_id('foo')
            vds2 = hc.import_vcf('src/test/resources/sample.vcf').with_id('bar')
            vds1.join(vds2).write('output/vds3.vds')

        the corresponding history file generated for `vds3.vds` will be as follows:

        .. code-block: python

            foo = hc.import_vcf('src/test/resources/sample.vcf')
            bar = hc.import_vcf('src/test/resources/sample.vcf')
            foo.join(bar).write('output/vds3.vds')

        If `with_id` is not used, the corresponding history file would be:

        .. code-block: python

            hc.import_vcf('src/test/resources/sample.vcf')
                .join(hc.import_vcf('src/test/resources/sample.vcf'))
                .write('/tmp/vds3.vds')

        :param id: Identifier for object.
        :type id: str

        :return: Input object.
        """

        self._history = self._history.set_varid(id)
        return self
