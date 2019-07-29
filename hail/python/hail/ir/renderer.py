from hail import ir
import abc
from typing import Sequence, List, Set, Dict


class Renderable(object):
    @abc.abstractmethod
    def render_head(self, r: 'Renderer') -> str:
        ...

    @abc.abstractmethod
    def render_tail(self, r: 'Renderer') -> str:
        ...

    @abc.abstractmethod
    def render_children(self, r: 'Renderer') -> Sequence['Renderable']:
        ...


class RenderableStr(Renderable):
    def __init__(self, s: str):
        self.s = s

    def render_head(self, r: 'Renderer') -> str:
        return self.s

    @abc.abstractmethod
    def render_tail(self, r: 'Renderer') -> str:
        return ''

    @abc.abstractmethod
    def render_children(self, r: 'Renderer') -> Sequence['Renderable']:
        return []


class ParensRenderer(Renderable):
    def __init__(self, rs: Sequence['Renderable']):
        self.rs = rs

    def render_head(self, r: 'Renderer') -> str:
        return '('

    @abc.abstractmethod
    def render_tail(self, r: 'Renderer') -> str:
        return ')'

    @abc.abstractmethod
    def render_children(self, r: 'Renderer') -> Sequence['Renderable']:
        return self.rs


class RenderableQueue(object):
    def __init__(self, elements: Sequence[Renderable], tail: str):
        self._elements = elements
        self._elements_len = len(elements)
        self.tail = tail
        self._idx = 0

    def exhausted(self):
        return self._elements_len == self._idx

    def pop(self) -> Renderable:
        idx = self._idx
        self._idx += 1
        return self._elements[idx]


class RQStack(object):
    def __init__(self):
        self._list: List[RenderableQueue] = []
        self._idx = -1

    def push(self, x: RenderableQueue):
        self._idx += 1
        self._list.append(x)

    def peek(self) -> RenderableQueue:
        return self._list[self._idx]

    def pop(self) -> RenderableQueue:
        self._idx -= 1
        return self._list.pop()

    def non_empty(self) -> bool:
        return self._idx >= 0

    def is_empty(self) -> bool:
        return self._idx < 0


class Renderer(object):
    def __init__(self, stop_at_jir=False):
        self.stop_at_jir = stop_at_jir
        self.count = 0
        self.jirs = {}

    def add_jir(self, jir):
        jir_id = f'm{self.count}'
        self.count += 1
        self.jirs[jir_id] = jir
        return jir_id

    def __call__(self, x: 'Renderable'):
        stack = RQStack()
        builder = []

        while x is not None or stack.non_empty():
            if x is not None:
                # TODO: it would be nice to put the JavaIR logic in BaseIR somewhere but this isn't trivial
                if self.stop_at_jir and hasattr(x, '_jir'):
                    jir_id = self.add_jir(x._jir)
                    if isinstance(x, ir.MatrixIR):
                        builder.append(f'(JavaMatrix {jir_id})')
                    elif isinstance(x, ir.TableIR):
                        builder.append(f'(JavaTable {jir_id})')
                    elif isinstance(x, ir.BlockMatrixIR):
                        builder.append(f'(JavaBlockMatrix {jir_id})')
                    else:
                        assert isinstance(x, ir.IR)
                        builder.append(f'(JavaIR {jir_id})')
                else:
                    head = x.render_head(self)
                    if head != '':
                        builder.append(x.render_head(self))
                    stack.push(RenderableQueue(x.render_children(self), x.render_tail(self)))
                x = None
            else:
                top = stack.peek()
                if top.exhausted():
                    stack.pop()
                    builder.append(top.tail)
                else:
                    builder.append(' ')
                    x = top.pop()

        return ''.join(builder)


class Scope:
    def __init__(self):
        self.visited: Set[int] = set()
        self.lifted_lets: Dict[int, str] = {}
        self.let_bodies: List[str] = []

    def visit(self, x: 'BaseIR'):
        self.visited.add(id(x))

class CSERenderer(Renderer):
    def __init__(self, stop_at_jir=False):
        self.stop_at_jir = stop_at_jir
        self.jir_count = 0
        self.jirs = {}
        self.memo: Dict[int, Sequence[str]] = {}
        self.uid_count = 0
        self.scopes: Dict[int, Scope] = {}

    def uid(self) -> str:
        self.uid_count += 1
        return f'__cse_{self.uid_count}'

    def add_jir(self, x: 'ir.BaseIR'):
        jir_id = f'm{self.jir_count}'
        self.jir_count += 1
        self.jirs[jir_id] = x._jir
        if isinstance(x, ir.MatrixIR):
            return f'(JavaMatrix {jir_id})'
        elif isinstance(x, ir.TableIR):
            return f'(JavaTable {jir_id})'
        elif isinstance(x, ir.BlockMatrixIR):
            return f'(JavaBlockMatrix {jir_id})'
        else:
            assert isinstance(x, ir.IR)
            return f'(JavaIR {jir_id})'

    def find_in_scope(self, x: 'ir.BaseIR', context: List[Scope]) -> int:
        for i in range(len(context)):
            if id(x) in context[i].visited:
                return i
        return -1

    def lifted_in_scope(self, x: 'ir.BaseIR', context: List[Scope]) -> int:
        for i in range(len(context)):
            if id(x) in context[i].lifted_lets:
                return i
        return -1

    Context = (Dict[str, int], Dict[str, int], Dict[str, int])

    # Pre:
    # * 'context' is a list of 'Scope's, one for each potential let-insertion
    #   site.
    # * 'ref_to_scope' maps each bound variable to the index in 'context' of the
    #   scope of its binding site.
    # Post:
    # * Returns set of free variables in 'x'.
    # * Each subtree of 'x' is flagged as visited in the outermost scope
    #   containing all of its free variables.
    # * Each subtree previously visited (either an earlier subtree of 'x', or
    #   marked visited in 'context') is added to set of lets in its (previously
    #   computed) outermost scope.
    # * 'self.scopes' is updated to map subtrees y of 'x' to scopes containing
    #   any lets to be inserted above y.
    def recur(self, scopes: List[Scope], context: Context, x: 'ir.BaseIR') -> Set[str]:
        # Ref nodes should never be lifted to a let. (Not that it would be
        # incorrect, just pointlessly adding names for the same thing.)
        if isinstance(x, ir.Ref):
            return {x.name}
        if isinstance(x, ir.GetField):
            print('...')
        free_vars = set()
        for i in range(len(x.children)):
            child = x.children[i]
            # FIXME: maintain a union of seen nodes in all scopes
            seen_in_scope = self.find_in_scope(child, scopes)
            if seen_in_scope >= 0:
                # we've seen 'child' before, no need to traverse
                if id(child) not in scopes[seen_in_scope].lifted_lets and isinstance(child, ir.IR):
                    # second time we've seen 'child', lift to a let
                    scopes[seen_in_scope].lifted_lets[id(child)] = self.uid()
            elif self.stop_at_jir and hasattr(child, '_jir'):
                self.memo[id(child)] = self.add_jir(child)
            else:
                if x.binds(i) or x.new_block(i):
                    def get_vars(bindings):
                        if isinstance(bindings, dict):
                            bindings = bindings.items()
                        return [var for (var, _) in bindings]
                    eval_b = get_vars(x.bindings(i))
                    agg_b = get_vars(x.agg_bindings(i))
                    scan_b = get_vars(x.scan_bindings(i))
                    new_scope = Scope()
                    if x.new_block(i):
                        # Repeated subtrees of this child should never be lifted to
                        # lets above 'x'. We accomplish that by clearing the context
                        # in the recursive call.
                        child_scopes = [new_scope]
                        (eval_c, agg_c, scan_c) = ({}, {}, {})
                        new_idx = 0
                    else:
                        new_idx = len(scopes)
                        scopes.append(new_scope)
                        child_scopes = scopes
                        (eval_c, agg_c, scan_c) = x.child_context_without_bindings(i, context)
                    eval_c = ir.base_ir._env_bind(eval_c, *[(var, new_idx) for var in eval_b])
                    agg_c = ir.base_ir._env_bind(agg_c, *[(var, new_idx) for var in agg_b])
                    scan_c = ir.base_ir._env_bind(scan_c, *[(var, new_idx) for var in scan_b])
                    child_free_vars = self.recur(child_scopes, (eval_c, agg_c, scan_c), child)
                    child_free_vars.difference_update(eval_b, agg_b, scan_b)
                    free_vars |= child_free_vars
                    if not x.new_block(i):
                        scopes.pop()
                    new_scope.visited.clear()
                    self.scopes[id(child)] = new_scope
                else:
                    free_vars |= self.recur(scopes, x.child_context_without_bindings(i, context), child)
        if len(free_vars) > 0:
            outermost_scope = max((context[0].get(v, 0) for v in free_vars))
        else:
            outermost_scope = 0
        scopes[outermost_scope].visited.add(id(x))
        return free_vars

    def print(self, builder: List[str], context: List[Scope], x: Renderable):
        if id(x) in self.memo:
            builder.append(self.memo[id(x)])
            return
        insert_lets = id(x) in self.scopes and len(self.scopes[id(x)].lifted_lets) > 0
        if insert_lets:
            local_builder = []
            context.append(self.scopes[id(x)])
        else:
            local_builder = builder
        head = x.render_head(self)
        if head != '':
            local_builder.append(head)
        children = x.render_children(self)
        for i in range(0, len(children)):
            local_builder.append(' ')
            child = children[i]
            lift_to = self.lifted_in_scope(child, context)
            if lift_to >= 0:
                name = context[lift_to].lifted_lets[id(child)]
                if id(child) not in context[lift_to].visited:
                    context[lift_to].visited.add(id(child))
                    let_body = [f'(Let {name} ']
                    self.print(let_body, context, child)
                    let_body.append(' ')
                    # let_bodies is built post-order, which guarantees earlier
                    # lets can't refer to later lets
                    context[lift_to].let_bodies.append(let_body)
                local_builder.append(f'(Ref {name})')
            else:
                self.print(local_builder, context, child)
        local_builder.append(x.render_tail(self))
        if insert_lets:
            context.pop()
            for let_body in self.scopes[id(x)].let_bodies:
                builder.extend(let_body)
            builder.extend(local_builder)
            num_lets = len(self.scopes[id(x)].lifted_lets)
            for i in range(num_lets):
                builder.append(')')


    def __call__(self, x: 'BaseIR') -> str:
        x.typ
        root_scope = Scope()
        free_vars = self.recur([root_scope], ({}, {}, {}), x)
        root_scope.visited = set()
        if len(free_vars) != 0:
            print('...')
        assert(len(free_vars) == 0)
        self.scopes[id(x)] = root_scope
        builder = []
        self.print(builder, [], x)
        return ''.join(builder)
