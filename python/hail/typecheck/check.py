from decorator import decorator, getargspec
from types import ClassType, NoneType
import re


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


class LambdaChecker(TypeChecker):
    def __init__(self, f, s):
        self.f = f
        self.s = s
        super(LambdaChecker, self).__init__()

    def check(self, x):
        return self.f(x)

    def expects(self):
        return self.s


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
        any_pass = False
        for c in self.checkers:
            any_pass = any_pass or c.check(x)
        return any_pass

    def expects(self):
        return '(' + ' or '.join([c.expects() for c in self.checkers]) + ')'


class CollectionChecker(TypeChecker):
    def __init__(self, collection_checker, element_checker):
        self.cc = collection_checker
        self.ec = element_checker
        super(CollectionChecker, self).__init__()

    def check(self, x):
        passes = self.cc.check(x)
        if passes:
            for elt in x:
                passes = passes and self.ec.check(elt)
        return passes

    def expects(self):
        return '%s[%s]' % (self.cc.expects(), self.ec.expects())


class DictChecker(TypeChecker):
    def __init__(self, key_checker, value_checker):
        self.kc = key_checker
        self.vc = value_checker
        super(DictChecker, self).__init__()

    def check(self, x):
        passes = isinstance(x, dict)
        if passes:
            for k, v in x.iteritems():
                passes = passes and self.kc.check(k)
                passes = passes and self.vc.check(v)
        return passes

    def expects(self):
        return 'dict[%s, %s]' % (self.kc.expects(), self.vc.expects())


def only(t):
    if isinstance(t, type) or type(t) is ClassType:
        return LambdaChecker(lambda x: isinstance(x, t), extract(t))
    elif isinstance(t, TypeChecker):
        return t
    else:
        raise RuntimeError("invalid typecheck signature: expected 'type' or 'TypeChecker', found '%s'" % type(t))


def oneof(*args):
    return MultipleTypeChecker([only(x) for x in args])


def nullable(t):
    return oneof(t, NoneType)


def listof(t):
    return CollectionChecker(only(list), only(t))


def dictof(k, v):
    return DictChecker(only(k), only(v))


none = only(NoneType)

anytype = LambdaChecker(lambda x: True, 'Any')

strlike = oneof(str, unicode)

integral = oneof(int, long)

numeric = oneof(int, long, float)

def check_all(name, args, spec, checks):
    assert len(args) == len(spec)

    # ensure that the typecheck signature is appropriate and matches the function signature
    if set(spec) != set(checks.keys()):
        unmatched_tc = [k for k in checks if k not in spec]
        unmatched_f = [k for k in spec if k not in checks]
        if unmatched_f or unmatched_tc:
            msg = ''
            if unmatched_tc:
                msg += 'unmatched typecheck arguments: [ %s ]' % \
                       ', '.join(["'%s'" % k for k in unmatched_tc])
            if unmatched_f:
                if msg:
                    msg += ', and '
                msg += 'function parameters with no defined type: [ %s ]' % \
                       ', '.join(["'%s'" % k for k in unmatched_f])
            raise RuntimeError('%s: invalid typecheck signature: %s' % (name, msg))

    for i, arg in enumerate(args):
        argname = spec[i]

        tc = only(checks[argname])

        if not tc.check(arg):
            raise TypeError("%s: parameter '%s': expected %s, found %s: '%s'" %
                            (name, argname, tc.expects(), extract(type(arg)), str(arg)))



def typecheck_method(**kw):
    def _typecheck(f, *args, **kwargs):
        argspec = getargspec(f)

        if not len(args) > 0 and isinstance(args[0], ClassType):
            raise RuntimeError('%s: no class found as first argument. Use typecheck instead of typecheck_method' %
                               f.__name__)

        if argspec.varkw or argspec.varargs:
            raise RuntimeError('%s: cannot typecheck methods that accept var args or var kwargs' % f.__name__)

        check_all(f.__name__, args[1:], argspec.args[1:], kw)

        return f(*args, **kwargs)

    return decorator(_typecheck)

def typecheck(**kw):

    def _typecheck(f, *args, **kwargs):
        argspec = getargspec(f)

        if argspec.varkw or argspec.varargs:
            raise RuntimeError('%s: cannot typecheck functions that accept var args or var keyword args' % f.__name__)

        check_all(f.__name__, args, argspec.args, kw)

        return f(*args, **kwargs)

    return decorator(_typecheck)
