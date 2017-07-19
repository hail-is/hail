import sys
from decorator import decorator, getargspec
import re
from hail.py3_compat import *


if sys.version_info > (3,):
    class_types = type
else:
    from types import ClassType
    class_types = (ClassType, type)


def extract(t):
    m = re.match("<(type|class) '(.*)'>", str(t))
    if m:
        return m.groups()[1]
    else:
        return str(t)


class TypeChecker(object):
    def __init__(self):
        pass

    def check(self, x):
        raise NotImplementedError

    def expects(self):
        raise NotImplementedError


class MultipleTypeChecker(TypeChecker):
    def __init__(self, checkers):
        flat_checkers = []
        for c in checkers:
            if isinstance(c, MultipleTypeChecker):
                for cc in c.checkers:
                    flat_checkers.append(cc)
            else:
                flat_checkers.append(c)
        self.checkers = flat_checkers
        super(MultipleTypeChecker, self).__init__()

    def check(self, x):
        return any(c.check(x) for c in self.checkers)

    def expects(self):
        return '(' + ' or '.join([c.expects() for c in self.checkers]) + ')'


class CollectionChecker(TypeChecker):
    def __init__(self, collection_checker, element_checker):
        self.cc = collection_checker
        self.ec = element_checker
        super(CollectionChecker, self).__init__()

    def check(self, x):
        passes = self.cc.check(x)
        return passes and all(self.ec.check(elt) for elt in x)

    def expects(self):
        return '%s[%s]' % (self.cc.expects(), self.ec.expects())


class DictChecker(TypeChecker):
    def __init__(self, key_checker, value_checker):
        self.kc = key_checker
        self.vc = value_checker
        super(DictChecker, self).__init__()

    def check(self, x):
        passes = isinstance(x, dict)
        return passes and all(self.kc.check(k) and self.vc.check(v) for k, v in x.items())

    def expects(self):
        return 'dict[%s, %s]' % (self.kc.expects(), self.vc.expects())


class AnyChecker(TypeChecker):
    def __init__(self):
        super(AnyChecker, self).__init__()

    def check(self, x):
        return True

    def expects(self):
        return 'any'

class CharChecker(TypeChecker):
    def __init__(self):
        super(CharChecker, self).__init__()

    def check(self, x):
        return (isinstance(x, str) or isinstance(x, unicode)) and len(x) == 1

    def expects(self):
        return 'char'

class LiteralChecker(TypeChecker):
    def __init__(self, t):
        self.t = t
        super(LiteralChecker, self).__init__()

    def check(self, x):
        return isinstance(x, self.t)

    def expects(self):
        return extract(self.t)


def only(t):
    if isinstance(t, class_types):
        return LiteralChecker(t)
    elif isinstance(t, TypeChecker):
        return t
    else:
        raise RuntimeError("invalid typecheck signature: expected 'type' or 'TypeChecker', found '%s'" % type(t))


def oneof(*args):
    return MultipleTypeChecker([only(x) for x in args])


def listof(t):
    return CollectionChecker(only(list), only(t))


def tupleof(t):
    return CollectionChecker(only(tuple), only(t))


def dictof(k, v):
    return DictChecker(only(k), only(v))


none = only(type(None))


def nullable(t):
    return oneof(t, none)


anytype = AnyChecker()

strlike = oneof(str, unicode)

integral = oneof(int, long)

numeric = oneof(int, long, float)

char = CharChecker()

def check_all(f, args, kwargs, checks, is_method):
    spec = getargspec(f)
    name = f.__name__

    arg_names = []
    args_to_check = []

    # strip the first argument if is_method is true (this is the self parameter)
    if is_method:
        if not (len(args) > 0 and isinstance(args[0], object)):
            raise RuntimeError(
                '%s: no class found as first argument. Use typecheck instead of typecheck_method?' % name)
        arg_names.extend(spec.args[1:])
        args_to_check.extend(args[1:])
    else:
        arg_names.extend(spec.args)
        args_to_check.extend(args)

    # if f has varargs, tuple any unnamed args and pass that as a regular argument to the checker
    if spec.varargs:
        n_named_args = len(spec.args) - (1 if is_method else 0)
        tupled_varargs = tuple(args_to_check[n_named_args:])
        args_to_check = args_to_check[:n_named_args]
        args_to_check.append(tupled_varargs)
        arg_names.append(spec.varargs)

    # if f has varkw, pass them as a dict to the checker.
    if spec.varkw:
        args_to_check.append(kwargs)
        arg_names.append(spec.varkw)

    # ensure that the typecheck signature is appropriate and matches the function signature
    if set(arg_names) != set(checks.keys()):
        unmatched_tc = [k for k in checks if k not in arg_names]
        unmatched_f = [k for k in arg_names if k not in checks]
        if unmatched_f or unmatched_tc:
            msg = ''
            if unmatched_tc:
                msg += 'unmatched typecheck arguments: %s' % unmatched_tc
            if unmatched_f:
                if msg:
                    msg += ', and '
                msg += 'function parameters with no defined type: %s' % unmatched_f
            raise RuntimeError('%s: invalid typecheck signature: %s' % (name, msg))

    # type check the function arguments
    for argname, arg in zip(arg_names, args_to_check):
        tc = only(checks[argname])

        if not tc.check(arg):
            raise TypeError("%s: parameter '%s': expected %s, found %s: '%s'" %
                            (name, argname, tc.expects(), extract(type(arg)), str(arg)))


def typecheck_method(**checkers):
    def _typecheck(f, *args, **kwargs):
        check_all(f, args, kwargs, checkers, is_method=True)
        return f(*args, **kwargs)

    return decorator(_typecheck)


def typecheck(**checkers):
    def _typecheck(f, *args, **kwargs):
        check_all(f, args, kwargs, checkers, is_method=False)
        return f(*args, **kwargs)

    return decorator(_typecheck)
