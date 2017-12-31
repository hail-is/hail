from hail.typecheck import *

asttype = lazy()


class AST(object):
    @typecheck_method(children=tupleof(asttype))
    def __init__(self, *children):
        self.children = children

    def to_hql(self):
        pass

    def expand(self):
        return self.search(lambda _: True)

    def search(self, f, l=None):
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

asttype.set(AST)


class Reference(AST):
    @typecheck_method(name=strlike, top_level=bool)
    def __init__(self, name, top_level=False):
        self.name = name
        self.top_level = top_level
        super(Reference, self).__init__()

    def to_hql(self):
        return '`{}`'.format(self.name)


class UnaryOperation(AST):
    @typecheck_method(parent=AST, operation=strlike)
    def __init__(self, parent, operation):
        self.parent = parent
        self.operation = operation
        super(UnaryOperation, self).__init__(parent)

    def to_hql(self):
        return '({}({}))'.format(self.operation, self.parent.to_hql())


class BinaryOperation(AST):
    @typecheck_method(left=AST, right=AST, operation=strlike)
    def __init__(self, left, right, operation):
        self.left = left
        self.right = right
        self.operation = operation
        super(BinaryOperation, self).__init__(left, right)

    def to_hql(self):
        return '({} {} {})'.format(self.left.to_hql(), self.operation, self.right.to_hql())


class Select(AST):
    @typecheck_method(parent=AST, selection=strlike)
    def __init__(self, parent, selection):
        self.parent = parent
        self.selection = selection
        super(Select, self).__init__(parent)

        # TODO: create nested selection option

    def to_hql(self):
        return '{}.{}'.format(self.parent.to_hql(), self.selection)


class ApplyMethod(AST):
    @typecheck_method(method=strlike, args=tupleof(AST))
    def __init__(self, method, *args):
        self.method = method
        self.args = args
        super(ApplyMethod, self).__init__(*args)

    def to_hql(self):
        return '{}({})'.format(self.method, ', '.join(ast.to_hql() for ast in self.args))


class ClassMethod(AST):
    @typecheck_method(method=strlike, callee=AST, args=tupleof(AST))
    def __init__(self, method, callee, *args):
        self.method = method
        self.callee = callee
        self.args = args
        super(ClassMethod, self).__init__(callee, *args)

    def to_hql(self):
        return '{}.{}({})'.format(self.callee.to_hql(), self.method, ', '.join(ast.to_hql() for ast in self.args))


class LambdaClassMethod(AST):
    @typecheck_method(method=strlike, lambda_var=strlike, callee=AST, rhs=AST, args=tupleof(AST))
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


class Index(AST):
    @typecheck_method(parent=AST, key=AST)
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        super(Index, self).__init__(parent, key)

    def to_hql(self):
        return '{}[{}]'.format(self.parent.to_hql(), self.key.to_hql())


class Literal(AST):
    @typecheck_method(value=strlike)
    def __init__(self, value):
        self.value = value
        super(Literal, self).__init__()

    def to_hql(self):
        return self.value


class ArrayDeclaration(AST):
    @typecheck_method(values=listof(AST))
    def __init__(self, values):
        self.values = values
        super(ArrayDeclaration, self).__init__(*values)

    def to_hql(self):
        return '[ {} ]'.format(', '.join(c.to_hql() for c in self.values))


class StructDeclaration(AST):
    @typecheck_method(keys=listof(strlike), values=listof(AST))
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values
        super(StructDeclaration, self).__init__(*values)

    def to_hql(self):
        return '{' + ', '.join('`{}`: {}'.format(k, v.to_hql()) for k, v in zip(self.keys, self.values)) + '}'


class StructOp(AST):
    @typecheck_method(operation=strlike, parent=AST, keys=tupleof(strlike))
    def __init__(self, operation, parent, *keys):
        self.operation = operation
        self.parent = parent
        self.keys = keys
        super(StructOp, self).__init__(parent)

    def to_hql(self):
        return '{}({}, {})'.format(self.operation,
                                   self.parent.to_hql(),
                                   ', '.join('`{}`'.format(x) for x in self.keys))


class Condition(AST):
    @typecheck_method(predicate=AST, branch1=AST, branch2=AST)
    def __init__(self, predicate, branch1, branch2):
        self.predicate = predicate
        self.branch1 = branch1
        self.branch2 = branch2
        super(Condition, self).__init__(predicate, branch1, branch2)

    def to_hql(self):
        return 'if ({p}) {b1} else {b2}'.format(p=self.predicate.to_hql(),
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
    @typecheck_method(uid=strlike, definition=AST, expression=AST)
    def __init__(self, uid, definition, expression):
        self.uid = uid
        self.definition = definition
        self.expression = expression
        super(Bind, self).__init__(definition, expression)

    def to_hql(self):
        return "let {uid} = {left_expr} in {right_expr}".format(
            uid=self.uid,
            left_expr=self.definition.to_hql(),
            right_expr=self.expression.to_hql()
        )


class RegexMatch(AST):
    @typecheck_method(string=AST, regex=strlike)
    def __init__(self, string, regex):
        self.string = string
        self.regex = regex
        super(RegexMatch, self).__init__(string)

    def to_hql(self):
        return '("{regex}" ~ {string})'.format(regex=self.regex, string=self.string.to_hql())


class AggregableReference(AST):
    def __init__(self):
        self.is_set = False
        super(AggregableReference, self).__init__()

    @typecheck_method(identifier=strlike)
    def set(self, identifier):
        assert not self.is_set
        self.is_set = True
        self.identifier = identifier

    def to_hql(self):
        assert self.is_set
        return self.identifier

class GlobalJoinReference(AST):
    def __init__(self, uid):
        self.is_set = False
        self.uid = uid
        super(GlobalJoinReference, self).__init__()

    def set(self, target):
        from hail.api2.matrixtable import MatrixTable
        self.is_set = True
        if isinstance(target, MatrixTable):
            self.is_matrix = True
        else:
            self.is_matrix = False

    def to_hql(self):
        assert self.is_set
        if self.is_matrix:
            return 'global.`{}`'.format(self.uid)
        else:
            return self.uid


def rewrite_global_refs(ast, target):
    for a in ast.search(lambda a: isinstance(a, GlobalJoinReference)):
        a.set(target)

@typecheck(ast=AST, identifier=strlike)
def replace_aggregables(ast, identifier):
    for a in ast.search(lambda a: isinstance(a, AggregableReference)):
        a.set(identifier)
