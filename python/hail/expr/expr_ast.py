from hail.typecheck import *
from hail.utils.java import Env, escape_str, escape_id
from .expressions.indices import Indices
from typing import *
import abc
import hail

asttype = lazy()


class AST(object):
    @typecheck_method(children=asttype)
    def __init__(self, *children):
        self.children = children

    def to_hql(self):
        pass

    def expand(self):
        return self.search(lambda _: True)

    def search(self, f, l=None) -> Tuple['AST']:
        """
        Recursively searches the AST for nodes matching some pattern.
        """
        if l is None:
            l = []

        if (f(self)):
            l.append(self)
        for c in self.children:
            c.search(f, l)
        return tuple(l)

    @property
    def is_nested_field(self):
        return False


asttype.set(AST)


class _Reference(AST):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name
        super(_Reference, self).__init__()

    def to_hql(self):
        return escape_id(self.name)


class VariableReference(_Reference):
    @typecheck_method(name=str)
    def __init__(self, name: str):
        super(VariableReference, self).__init__(name)


class TopLevelReference(_Reference):
    def __init__(self, name: str, indices: Indices):
        self.indices = indices
        super(TopLevelReference, self).__init__(name)

    @property
    def is_nested_field(self):
        return True


class UnaryOperation(AST):
    @typecheck_method(parent=AST, operation=str)
    def __init__(self, parent, operation):
        self.parent = parent
        self.operation = operation
        super(UnaryOperation, self).__init__(parent)

    def to_hql(self):
        return '({}({}))'.format(self.operation, self.parent.to_hql())


class BinaryOperation(AST):
    @typecheck_method(left=AST, right=AST, operation=str)
    def __init__(self, left, right, operation):
        self.left = left
        self.right = right
        self.operation = operation
        super(BinaryOperation, self).__init__(left, right)

    def to_hql(self):
        return '({} {} {})'.format(self.left.to_hql(), self.operation, self.right.to_hql())


class Select(AST):
    @typecheck_method(parent=AST, name=str)
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        super(Select, self).__init__(parent)

        # TODO: create nested selection option

    def to_hql(self):
        return '{}.{}'.format(self.parent.to_hql(), escape_id(self.name))

    @property
    def is_nested_field(self):
        return self.parent.is_nested_field


class ApplyMethod(AST):
    @typecheck_method(method=str, args=AST)
    def __init__(self, method, *args):
        self.method = method
        self.args = args
        super(ApplyMethod, self).__init__(*args)

    def to_hql(self):
        return '{}({})'.format(self.method, ', '.join(ast.to_hql() for ast in self.args))


class ClassMethod(AST):
    @typecheck_method(method=str, callee=AST, args=AST)
    def __init__(self, method, callee, *args):
        self.method = method
        self.callee = callee
        self.args = args
        super(ClassMethod, self).__init__(callee, *args)

    def to_hql(self):
        return '{}.{}({})'.format(self.callee.to_hql(), self.method, ', '.join(ast.to_hql() for ast in self.args))


class LambdaClassMethod(AST):
    @typecheck_method(method=str, lambda_var=str, callee=AST, rhs=AST, args=AST)
    def __init__(self, method, lambda_var, callee, rhs, *args):
        self.method = method
        self.lambda_var = lambda_var
        self.callee = callee
        self.rhs = rhs
        self.args = args
        super(LambdaClassMethod, self).__init__(callee, rhs, *args)

    def to_hql(self):
        if self.args:
            return '{}.{}({} => {}, {})'.format(self.callee.to_hql(), self.method, self.lambda_var, self.rhs.to_hql(),
                                                ', '.join(a.to_hql() for a in self.args))
        else:
            return '{}.{}({} => {})'.format(self.callee.to_hql(), self.method, self.lambda_var, self.rhs.to_hql())


class LambdaFunction(AST):
    @typecheck_method(method=str, lambda_var=str, rhs=AST, args=AST)
    def __init__(self, method, lambda_var, rhs, *args):
        self.method = method
        self.lambda_var = lambda_var
        self.rhs = rhs
        self.args = args
        super(LambdaFunction, self).__init__(rhs, *args)

    def to_hql(self):
        if self.args:
            return '{}({} => {}, {})'.format(self.method, self.lambda_var, self.rhs.to_hql(),
                                             ', '.join(a.to_hql() for a in self.args))
        else:
            return '{}({} => {})'.format(self.method, self.lambda_var, self.rhs.to_hql())


class Index(AST):
    @typecheck_method(parent=AST, key=AST)
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        super(Index, self).__init__(parent, key)

    def to_hql(self):
        return '{}[{}]'.format(self.parent.to_hql(), self.key.to_hql())


class Literal(AST):
    @typecheck_method(value=str)
    def __init__(self, value):
        self.value = value
        super(Literal, self).__init__()

    def to_hql(self):
        return '({})'.format(self.value)


class ArrayDeclaration(AST):
    @typecheck_method(values=sequenceof(AST))
    def __init__(self, values):
        self.values = values
        super(ArrayDeclaration, self).__init__(*values)

    def to_hql(self):
        return '[ {} ]'.format(', '.join(c.to_hql() for c in self.values))


class StructDeclaration(AST):
    @typecheck_method(keys=sequenceof(str), values=sequenceof(AST))
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values
        super(StructDeclaration, self).__init__(*values)

    def to_hql(self):
        return '{' + ', '.join('{}: {}'.format(escape_id(k), v.to_hql()) for k, v in zip(self.keys, self.values)) + '}'


class TupleDeclaration(AST):
    @typecheck_method(values=AST)
    def __init__(self, *values):
        self.values = values
        super(TupleDeclaration, self).__init__(*values)

    def to_hql(self):
        return 'Tuple(' + ', '.join(v.to_hql() for v in self.values) + ')'


class StructOp(AST):
    @typecheck_method(operation=str, parent=AST, keys=str)
    def __init__(self, operation, parent, *keys):
        self.operation = operation
        self.parent = parent
        self.keys = keys
        super(StructOp, self).__init__(parent)

    def to_hql(self):
        return '{}({}{})'.format(self.operation,
                                 self.parent.to_hql(),
                                 ''.join(', {}'.format(escape_id(x)) for x in self.keys))


class Condition(AST):
    @typecheck_method(predicate=AST, branch1=AST, branch2=AST)
    def __init__(self, predicate, branch1, branch2):
        self.predicate = predicate
        self.branch1 = branch1
        self.branch2 = branch2
        super(Condition, self).__init__(predicate, branch1, branch2)

    def to_hql(self):
        return '(if ({p}) {b1} else {b2})'.format(p=self.predicate.to_hql(),
                                                  b1=self.branch1.to_hql(),
                                                  b2=self.branch2.to_hql())


class Slice(AST):
    @typecheck_method(start=nullable(AST), stop=nullable(AST))
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        super(Slice, self).__init__(*[x for x in [start, stop] if x is not None])

    def to_hql(self):
        return "{start}:{end}".format(start=self.start.to_hql() if self.start else '',
                                      end=self.stop.to_hql() if self.stop else '')


class Bind(AST):
    @typecheck_method(uids=sequenceof(str), definitions=sequenceof(AST), expression=AST)
    def __init__(self, uids, definitions, expression):
        self.uids = uids
        self.definitions = definitions
        self.expression = expression
        super(Bind, self).__init__(*definitions, expression)

    def to_hql(self):
        bindings = ' and '.join(f'{uid} = {ast.to_hql()}' for uid, ast in zip(self.uids, self.definitions))
        return f'(let {bindings} in {self.expression.to_hql()})'


class RegexMatch(AST):
    @typecheck_method(string=AST, regex=str)
    def __init__(self, string, regex):
        self.string = string
        self.regex = regex
        super(RegexMatch, self).__init__(string)

    def to_hql(self):
        return '("{regex}" ~ {string})'.format(regex=escape_str(self.regex), string=self.string.to_hql())


class AggregableReference(AST):
    def __init__(self):
        super(AggregableReference, self).__init__()

    def to_hql(self):
        return 'AGG'


class Join(AST):
    _idx = 0
    def __init__(self,
                 virtual_ast: AST,
                 temp_vars: Sequence[str],
                 join_exprs: Sequence[Any],
                 join_func: Callable):
        super(Join, self).__init__(*(e._ast for e in join_exprs))
        self.virtual_ast = virtual_ast
        self.temp_vars = temp_vars
        self.join_exprs = join_exprs
        self.join_func = join_func
        self.idx = Join._idx
        Join._idx += 1

    def to_hql(self):
        return self.virtual_ast.to_hql()


class Broadcast(AST):
    def __init__(self, value: Any, dtype: 'hail.HailType'):
        super(Broadcast, self).__init__()
        self.value = value
        self.dtype = dtype
        self.uid = Env.get_uid()

    def to_hql(self):
        return f'global.`{self.uid}`'
