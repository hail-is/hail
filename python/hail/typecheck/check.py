from decorator import decorator, getargspec
from types import ClassType, NoneType, InstanceType
import re


class TypecheckFailure(Exception):
    pass


identity = lambda x: x


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
        for tc in self.checkers:
            try:
                return tc.check(x)
            except TypecheckFailure:
                pass
        raise TypecheckFailure()

    def expects(self):
        return '(' + ' or '.join([c.expects() for c in self.checkers]) + ')'


class ListChecker(TypeChecker):
    def __init__(self, element_checker):
        self.ec = element_checker
        super(ListChecker, self).__init__()

    def check(self, x):
        if not isinstance(x, list):
            raise TypecheckFailure
        x_ = []
        tc = self.ec
        for elt in x:
            elt_ = tc.check(elt)
            x_.append(elt_)
        return x_

    def expects(self):
        return 'list[%s]' % (self.ec.expects())


class SetChecker(TypeChecker):
    def __init__(self, element_checker):
        self.ec = element_checker
        super(SetChecker, self).__init__()

    def check(self, x):
        if not isinstance(x, set):
            raise TypecheckFailure
        x_ = set()
        tc = self.ec
        for elt in x:
            elt_ = tc.check(elt)
            x_.add(elt_)
        return x_

    def expects(self):
        return 'set[%s]' % (self.ec.expects())


class TupleChecker(TypeChecker):
    def __init__(self, element_checker):
        self.ec = element_checker
        super(TupleChecker, self).__init__()

    def check(self, x):
        if not isinstance(x, tuple):
            raise TypecheckFailure
        x_ = []
        tc = self.ec
        for elt in x:
            elt_ = tc.check(elt)
            x_.append(elt_)
        return tuple(x_)

    def expects(self):
        return 'tuple[%s]' % (self.ec.expects())


class DictChecker(TypeChecker):
    def __init__(self, key_checker, value_checker):
        self.kc = key_checker
        self.vc = value_checker
        super(DictChecker, self).__init__()

    def check(self, x):
        if not isinstance(x, dict):
            raise TypecheckFailure
        x_ = {}
        kc = self.kc
        vc = self.vc
        for k, v in x.items():
            k_ = kc.check(k)
            v_ = vc.check(v)
            x_[k_] = v_
        return x_

    def expects(self):
        return 'dict[%s, %s]' % (self.kc.expects(), self.vc.expects())

    def coerce(self, x):
        kc = self.kc
        vc = self.vc
        return {kc.coerce(k): vc.coerce(v) for k, v in x}


class SizedTupleChecker(TypeChecker):
    def __init__(self, *elt_checkers):
        self.ec = elt_checkers
        self.n = len(elt_checkers)
        super(SizedTupleChecker, self).__init__()

    def check(self, x):
        if not isinstance(x, tuple):
            raise TypecheckFailure
        x_ = []
        for tc, elt in zip(self.ec, x):
            elt_ = tc.check(elt)
            x_.append(elt_)
        return tuple(x_)

    def expects(self):
        return 'tuple[' + ','.join(["{}".format(ec.expects()) for ec in self.ec]) + ']'


class AnyChecker(TypeChecker):
    def __init__(self):
        super(AnyChecker, self).__init__()

    def check(self, x):
        return x

    def expects(self):
        return 'any'


class CharChecker(TypeChecker):
    def __init__(self):
        super(CharChecker, self).__init__()

    def check(self, x):
        if (isinstance(x, str) or isinstance(x, unicode)) and len(x) == 1:
            return x
        else:
            raise TypecheckFailure

    def expects(self):
        return 'char'


class LiteralChecker(TypeChecker):
    def __init__(self, t):
        self.t = t
        super(LiteralChecker, self).__init__()

    def check(self, x):
        if isinstance(x, self.t):
            return x
        else:
            raise TypecheckFailure

    def expects(self):
        return extract(self.t)


class LazyChecker(TypeChecker):
    def __init__(self):
        self.t = None
        super(LazyChecker, self).__init__()

    def set(self, t):
        self.t = t

    def check(self, x):
        if not self.t:
            raise RuntimeError("LazyChecker not initialized. Use 'set' to provide the expected type")
        if isinstance(x, self.t):
            return x
        else:
            raise TypecheckFailure

    def expects(self):
        if not self.t:
            raise RuntimeError("LazyChecker not initialized. Use 'set' to provide the expected type")
        return extract(self.t)


class ExactlyTypeChecker(TypeChecker):
    def __init__(self, v):
        self.v = v
        super(ExactlyTypeChecker, self).__init__()

    def check(self, x):
        if x == self.v:
            return x
        else:
            raise TypecheckFailure

    def expects(self):
        return str(self.v)


class CoercionChecker(MultipleTypeChecker):
    """Type checker that performs argument transformations.

    The `fs` argument should be a varargs of 2-tuples which contain a
    TypeChecker and a lambda function, e.g.:

    ((only(int), lambda x: x * 2),
     listof(int), lambda x: x[0]))
    """

    def __init__(self, *fs):
        self.fs = fs
        super(CoercionChecker, self).__init__([c for c, _ in fs])

    def check(self, x):
        for tc, f in self.fs:
            try:
                return f(tc.check(x))
            except TypecheckFailure:
                pass
        raise TypecheckFailure


def only(t):
    if isinstance(t, type) or type(t) is ClassType:
        return LiteralChecker(t)
    elif isinstance(t, TypeChecker):
        return t
    else:
        raise RuntimeError("invalid typecheck signature: expected 'type' or 'TypeChecker', found '%s'" % type(t))


def exactly(v):
    return ExactlyTypeChecker(v)


def oneof(*args):
    return MultipleTypeChecker([only(x) for x in args])


def enumeration(*args):
    return MultipleTypeChecker([exactly(x) for x in args])


def nullable(t):
    return oneof(t, NoneType)


def listof(t):
    return ListChecker(only(t))


def tupleof(t):
    return TupleChecker(only(t))


def sized_tupleof(*args):
    return SizedTupleChecker(*[only(x) for x in args])


def setof(t):
    return SetChecker(only(t))


def dictof(k, v):
    return DictChecker(only(k), only(v))


def transformed(*tcs):
    fs = []
    for tc, f in tcs:
        tc = only(tc)
        fs.append((tc, f))
    return CoercionChecker(*fs)


def lazy():
    return LazyChecker()


none = only(NoneType)

anytype = AnyChecker()

strlike = oneof(str, unicode)

integral = oneof(int, long)

numeric = oneof(int, long, float)

char = CharChecker()


def check_all(f, args, kwargs, checks, is_method):
    spec = getargspec(f)
    name = f.__name__

    args_ = []

    # strip the first argument if is_method is true (this is the self parameter)
    if is_method:
        if not (len(args) > 0 and isinstance(args[0], object)):
            raise RuntimeError(
                '%s: no class found as first argument. Use typecheck instead of typecheck_method?' % name)
        named_args = spec.args[1:]
        pos_args = args[1:]
        args_.append(args[0])
    else:
        named_args = spec.args[:]
        pos_args = args[:]

    signature_namespace = set(named_args).union(
        set(filter(lambda x: x is not None, [spec.varargs, spec.varkw])))
    tc_namespace = set(checks.keys())

    # ensure that the typecheck signature is appropriate and matches the function signature
    if signature_namespace != tc_namespace:
        unmatched_tc = list(tc_namespace - signature_namespace)
        unmatched_sig = list(signature_namespace - tc_namespace)
        if unmatched_sig or unmatched_tc:
            msg = ''
            if unmatched_tc:
                msg += 'unmatched typecheck arguments: %s' % unmatched_tc
            if unmatched_sig:
                if msg:
                    msg += ', and '
                msg += 'function parameters with no defined type: %s' % unmatched_sig
            raise RuntimeError('%s: invalid typecheck signature: %s' % (name, msg))

    # if f has varargs, tuple any unnamed args and pass that as a regular argument to the checker
    for i in range(len(pos_args)):
        arg = pos_args[i]
        argname = named_args[i] if i < len(named_args) else spec.varargs
        tc = checks[argname]
        try:
            arg_ = tc.check(arg)
            args_.append(arg_)
        except TypecheckFailure:
            if i < len(named_args):
                raise TypeError("{fname}: parameter '{argname}': "
                                "expected {expected}, found {found}: '{arg}'".format(
                    fname=name,
                    argname=argname,
                    expected=tc.expects(),
                    found=extract(type(arg)),
                    arg=str(arg)
                ))
            else:
                raise TypeError("{fname}: parameter '*{argname}' (arg {idx} of {tot}): "
                                "expected {expected}, found {found}: '{arg}'".format(
                    fname=name,
                    argname=argname,
                    idx=i - len(named_args),
                    tot=len(pos_args) - len(named_args),
                    expected=tc.expects(),
                    found=extract(type(arg)),
                    arg=str(arg)
                ))

    kwargs_ = {}
    if spec.varkw:
        tc = checks[spec.varkw]
        for argname, arg in kwargs.items():
            try:
                arg_ = tc.check(arg)
                kwargs_[argname] = arg_
            except TypecheckFailure:
                raise TypeError("{fname}: keyword argument '{argname}': "
                                "expected {expected}, found {found}: '{arg}'".format(
                    fname=name,
                    argname=argname,
                    expected=tc.expects(),
                    found=extract(type(arg)),
                    arg=str(arg)
                ))

    return args_, kwargs_


def typecheck_method(**checkers):
    checkers = {k: only(v) for k, v in checkers.items()}

    def _typecheck(f, *args, **kwargs):
        args_, kwargs_ = check_all(f, args, kwargs, checkers, is_method=True)
        return f(*args_, **kwargs_)

    return decorator(_typecheck)


def typecheck(**checkers):
    checkers = {k: only(v) for k, v in checkers.items()}

    def _typecheck(f, *args, **kwargs):
        args_, kwargs_ = check_all(f, args, kwargs, checkers, is_method=False)
        return f(*args_, **kwargs_)

    return decorator(_typecheck)
