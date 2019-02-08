import re
import inspect
import abc
import collections
from decorator import decorator


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
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def check(self, x, caller, param):
        ...

    @abc.abstractmethod
    def expects(self):
        ...

    def format(self, arg):
        return f"{extract(type(arg))}: {arg}"


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

    def check(self, x, caller, param):
        for tc in self.checkers:
            try:
                return tc.check(x, caller, param)
            except TypecheckFailure:
                pass
        raise TypecheckFailure()

    def expects(self):
        return '(' + ' or '.join([c.expects() for c in self.checkers]) + ')'


class SequenceChecker(TypeChecker):
    def __init__(self, element_checker):
        self.ec = element_checker
        super(SequenceChecker, self).__init__()

    def check(self, x, caller, param):
        # reject str because of errors due to sequenceof(strlike) permitting str
        if not isinstance(x, collections.Sequence) or isinstance(x, str):
            raise TypecheckFailure
        x_ = []
        tc = self.ec
        for elt in x:
            elt_ = tc.check(elt, caller, param)
            x_.append(elt_)
        return x_

    def expects(self):
        return 'Sequence[%s]' % (self.ec.expects())


class SetChecker(TypeChecker):
    def __init__(self, element_checker):
        self.ec = element_checker
        super(SetChecker, self).__init__()

    def check(self, x, caller, param):
        if not isinstance(x, set):
            raise TypecheckFailure
        x_ = set()
        tc = self.ec
        for elt in x:
            elt_ = tc.check(elt, caller, param)
            x_.add(elt_)
        return x_

    def expects(self):
        return 'set[%s]' % (self.ec.expects())


class TupleChecker(TypeChecker):
    def __init__(self, element_checker):
        self.ec = element_checker
        super(TupleChecker, self).__init__()

    def check(self, x, caller, param):
        if not isinstance(x, tuple):
            raise TypecheckFailure
        x_ = []
        tc = self.ec
        for elt in x:
            elt_ = tc.check(elt, caller, param)
            x_.append(elt_)
        return tuple(x_)

    def expects(self):
        return 'tuple[%s]' % (self.ec.expects())


class DictChecker(TypeChecker):
    def __init__(self, key_checker, value_checker):
        self.kc = key_checker
        self.vc = value_checker
        super(DictChecker, self).__init__()

    def check(self, x, caller, param):
        if not isinstance(x, collections.abc.Mapping):
            raise TypecheckFailure
        x_ = {}
        kc = self.kc
        vc = self.vc
        for k, v in x.items():
            k_ = kc.check(k, caller, param)
            v_ = vc.check(v, caller, param)
            x_[k_] = v_
        return x_

    def expects(self):
        return 'Mapping[%s, %s]' % (self.kc.expects(), self.vc.expects())

    def coerce(self, x):
        kc = self.kc
        vc = self.vc
        return {kc.coerce(k): vc.coerce(v) for k, v in x}


class SizedTupleChecker(TypeChecker):
    def __init__(self, *elt_checkers):
        self.ec = elt_checkers
        self.n = len(elt_checkers)
        super(SizedTupleChecker, self).__init__()

    def check(self, x, caller, param):
        if not (isinstance(x, tuple) and len(x) == len(self.ec)):
            raise TypecheckFailure
        x_ = []
        for tc, elt in zip(self.ec, x):
            elt_ = tc.check(elt, caller, param)
            x_.append(elt_)
        return tuple(x_)

    def expects(self):
        return 'tuple[' + ','.join(["{}".format(ec.expects()) for ec in self.ec]) + ']'


class LinkedListChecker(TypeChecker):
    def __init__(self, type):
        self.type = type
        super(LinkedListChecker, self).__init__()

    def check(self, x, caller, param):
        from hail.utils import LinkedList
        if not isinstance(x, LinkedList):
            raise TypecheckFailure
        if x.type is not self.type:
            raise TypecheckFailure
        return x

    def expects(self):
        return 'linkedlist[%s]' % self.type


class AnyChecker(TypeChecker):
    def __init__(self):
        super(AnyChecker, self).__init__()

    def check(self, x, caller, param):
        return x

    def expects(self):
        return 'any'


class CharChecker(TypeChecker):
    def __init__(self):
        super(CharChecker, self).__init__()

    def check(self, x, caller, param):
        if isinstance(x, str) and len(x) == 1:
            return x
        else:
            raise TypecheckFailure

    def expects(self):
        return 'char'


class LiteralChecker(TypeChecker):
    def __init__(self, t):
        self.t = t
        super(LiteralChecker, self).__init__()

    def check(self, x, caller, param):
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

    def check(self, x, caller, param):
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
    def __init__(self, v, reference_equality=False):
        self.v = v
        self.reference_equality = reference_equality
        super(ExactlyTypeChecker, self).__init__()

    def check(self, x, caller, param):
        if self.reference_equality and x is self.v:
            return x
        elif not self.reference_equality and x == self.v:
            return x
        else:
            raise TypecheckFailure

    def expects(self):
        return repr(self.v)


class CoercionChecker(TypeChecker):
    """Type checker that performs argument transformations.

    The `fs` argument should be a varargs of 2-tuples that each contain a
    TypeChecker and a lambda function, e.g.:

    ((only(int), lambda x: x * 2),
     sequenceof(int), lambda x: x[0]))
    """

    def __init__(self, *fs):
        self.fs = fs
        super(CoercionChecker, self).__init__()

    def check(self, x, caller, param):
        for tc, f in self.fs:
            try:
                return f(tc.check(x, caller, param))
            except TypecheckFailure:
                pass
        raise TypecheckFailure

    def expects(self):
        return '(' + ' or '.join([c.expects() for c, _ in self.fs]) + ')'


class AnyFuncChecker(TypeChecker):
    def __init__(self):
        super(AnyFuncChecker, self).__init__()

    def check(self, x, caller, param):
        if not callable(x):
            raise TypecheckFailure
        return x

    def expects(self):
        return 'function'


class FunctionChecker(TypeChecker):
    def __init__(self, nargs, ret_checker):
        self.nargs = nargs
        self.ret_checker = ret_checker
        super(FunctionChecker, self).__init__()

    def check(self, x, caller, param):
        if not callable(x):
            raise TypecheckFailure
        spec = inspect.signature(x)
        if not len(spec.parameters) == self.nargs:
            raise TypecheckFailure

        def f(*args):
            ret = x(*args)
            try:
                return self.ret_checker.check(ret, caller, param)
            except TypecheckFailure:
                raise TypeError("'{caller}': '{param}': expected return type {expected}, found {found}".format(
                    caller=caller,
                    param=param,
                    expected=self.ret_checker.expects(),
                    found=self.ret_checker.format(ret)
                ))

        return f

    def expects(self):
        return '{}-argument function'.format(self.nargs)

    def format(self, arg):
        if not callable(arg):
            return super(FunctionChecker, self).format(arg)
        spec = inspect.getfullargspec(arg)
        return '{}-argument function'.format(len(spec.args))


def only(t):
    if isinstance(t, type):
        return LiteralChecker(t)
    elif isinstance(t, TypeChecker):
        return t
    else:
        raise RuntimeError("invalid typecheck signature: expected 'type' or 'TypeChecker', found '%s'" % type(t))


def exactly(v, reference_equality=False):
    return ExactlyTypeChecker(v, reference_equality)


def oneof(*args):
    return MultipleTypeChecker([only(x) for x in args])


def enumeration(*args):
    return MultipleTypeChecker([exactly(x) for x in args])


def nullable(t):
    return oneof(exactly(None, reference_equality=True), t)


def sequenceof(t):
    return SequenceChecker(only(t))


def tupleof(t):
    return TupleChecker(only(t))


def sized_tupleof(*args):
    return SizedTupleChecker(*[only(x) for x in args])


def linked_list(t):
    return LinkedListChecker(t)


def setof(t):
    return SetChecker(only(t))


def dictof(k, v):
    return DictChecker(only(k), only(v))


def func_spec(n, tc):
    return FunctionChecker(n, only(tc))

anyfunc = AnyFuncChecker()

def transformed(*tcs):
    fs = []
    for tc, f in tcs:
        tc = only(tc)
        fs.append((tc, f))
    return CoercionChecker(*fs)


def lazy():
    return LazyChecker()


anytype = AnyChecker()

numeric = oneof(int, float)

char = CharChecker()

table_key_type = nullable(
    oneof(
        transformed((str, lambda x: [x])),
        sequenceof(str)))


def get_signature(f) -> inspect.Signature:
    if hasattr(f, '__memo'):
        return f.__memo
    else:
        signature = inspect.signature(f)
        f.__memo = signature
        return signature


def check_meta(f, checks, is_method):
    if hasattr(f, '__checked'):
        return
    else:
        spec = get_signature(f)
        params = list(spec.parameters)
        if is_method:
            params = params[1:]
        signature_namespace = set(params)
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
                raise RuntimeError('%s: invalid typecheck signature: %s' % (f.__name__, msg))
        f.__checked = True


def check_all(f, args, kwargs, checks, is_method):
    spec = get_signature(f)
    check_meta(f, checks, is_method)
    name = f.__name__
    arg_list = list(args)

    args_ = []
    kwargs_ = {}

    has_varargs = any(param.kind == param.VAR_POSITIONAL for param in spec.parameters.values())
    n_pos_args = len(
        list(filter(
            lambda p: p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD),
            spec.parameters.values())))
    if not has_varargs and len(args) > n_pos_args:
        raise TypeError(f"'{name}' takes {n_pos_args} positional arguments, found {len(args)}")

    for i, (arg_name, param) in enumerate(spec.parameters.items()):
        if i == 0 and is_method:
            if not isinstance(arg_list[0], object):
                raise RuntimeError("no class found as first argument. Did you mean to use 'typecheck' "
                                   "instead of 'typecheck_method'?")
            args_.append(args[i])
            continue
        checker = checks[arg_name]
        assert isinstance(param, inspect.Parameter)
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            try:
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                    # must be positional
                    if i < len(args):
                        arg = args[i]
                        args_.append(checker.check(arg, name, arg_name))
                    # passed as keyword
                    else:
                        if arg_name in kwargs:
                            arg = kwargs.pop(arg_name)
                        else:
                            if param.default is inspect._empty:
                                raise TypeError(f'Expected {n_pos_args} positional arguments, '
                                                f'found {len(args)}')
                            arg = param.default
                        args_.append(checker.check(arg, name, arg_name))
                else:
                    if arg_name in kwargs:
                        arg = kwargs.pop(arg_name)
                    else:
                        if param.default is inspect._empty:
                            raise TypeError(f"{name}() missing required keyword-only argument '{arg_name}'")
                        arg = param.default
                    kwargs_[arg_name] = checker.check(arg, name, arg_name)
            except TypecheckFailure as e:
                raise TypeError("{fname}: parameter '{argname}': "
                                "expected {expected}, found {found}".format(
                    fname=name,
                    argname=arg_name,
                    expected=checker.expects(),
                    found=checker.format(arg)
                )) from e
        elif param.kind == param.VAR_POSITIONAL:
            # consume the rest of the positional arguments
            varargs = args[i:]
            for j, arg in enumerate(varargs):
                try:
                    args_.append(checker.check(arg, name, arg_name))
                except TypecheckFailure as e:
                    raise TypeError("{fname}: parameter '*{argname}' (arg {idx} of {tot}): "
                                    "expected {expected}, found {found}".format(
                        fname=name,
                        argname=arg_name,
                        idx=j,
                        tot=len(varargs),
                        expected=checker.expects(),
                        found=checker.format(arg)
                    )) from e
        else:
            assert param.kind == param.VAR_KEYWORD
            # kwargs now holds all variable kwargs
            for kwarg_name, arg in kwargs.items():
                try:
                    kwargs_[kwarg_name] = checker.check(arg, name, arg_name)
                except TypecheckFailure as e:
                    raise TypeError("{fname}: keyword argument '{argname}': "
                                    "expected {expected}, found {found}".format(
                        fname=name,
                        argname=kwarg_name,
                        expected=checker.expects(),
                        found=checker.format(arg))) from e
    return args_, kwargs_


def typecheck_method(**checkers):
    return _make_dec(checkers, is_method=True)


def typecheck(**checkers):
    return _make_dec(checkers, is_method=False)


def _make_dec(checkers, is_method):
    checkers = {k: only(v) for k, v in checkers.items()}

    @decorator
    def wrapper(__original_func, *args, **kwargs):
        args_, kwargs_ = check_all(__original_func, args, kwargs, checkers, is_method=is_method)
        return __original_func(*args_, **kwargs_)

    return wrapper
