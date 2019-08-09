from hail import ir
import abc
from typing import Sequence, List, Set, Dict, Callable, Tuple


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


class BindingSite:
    def __init__(self, lifted_lets: Dict[int, Tuple[str, 'ir.BaseIR']], depth: int):
        self.depth = depth
        self.lifted_lets = lifted_lets


class PrintStackFrame:
    def __init__(self, binding_site: BindingSite):
        self.depth = binding_site.depth
        self.lifted_lets = binding_site.lifted_lets
        self.visited = {}
        self.let_bodies = []


Vars = Dict[str, int]
Context = (Vars, Vars, Vars)


class StackFrame:
    def __init__(self, min_binding_depth: int, context: Context, x: 'ir.BaseIR',
                 new_bindings=(None, None, None)):
        # immutable
        self.min_binding_depth = min_binding_depth
        self.context = context
        self.node = x
        (self._parent_eval_bindings, self._parent_agg_bindings,
         self._parent_scan_bindings) = new_bindings
        # mutable
        self.free_vars = {}
        self.visited: Dict[int, 'ir.BaseIR'] = {}
        self.lifted_lets: Dict[int, (str, 'ir.BaseIR')] = {}
        self.child_idx = 0

    # compute depth at which we might bind this node
    def bind_depth(self) -> int:
        if len(self.free_vars) > 0:
            bind_depth = max(self.free_vars.values())
            bind_depth = max(bind_depth, self.min_binding_depth)
        else:
            bind_depth = self.min_binding_depth
        return bind_depth

    def make_child_frame(self, depth: int) -> 'StackFrame':
        x = self.node
        i = self.child_idx - 1
        child = x.children[i]
        if x.new_block(i):
            child_outermost_scope = depth
        else:
            child_outermost_scope = self.min_binding_depth

        # compute vars bound in 'child' by 'node'
        def get_vars(bindings):
            return [var for (var, _) in bindings]
        eval_bindings = get_vars(x.bindings(i))
        agg_bindings = get_vars(x.agg_bindings(i))
        scan_bindings = get_vars(x.scan_bindings(i))

        child_context = x.child_context_without_bindings(i, self.context)
        if x.binds(i):
            (eval_c, agg_c, scan_c) = child_context
            eval_c = ir.base_ir._env_bind(eval_c, *[(var, depth) for var in
                                                    eval_bindings])
            agg_c = ir.base_ir._env_bind(agg_c, *[(var, depth) for var in
                                                  agg_bindings])
            scan_c = ir.base_ir._env_bind(scan_c, *[(var, depth) for var in
                                                    scan_bindings])
            child_context = (eval_c, agg_c, scan_c)

        new_bindings = (eval_bindings, agg_bindings, scan_bindings)

        return StackFrame(child_outermost_scope, child_context, child,
                          new_bindings)

    def update_parent_free_vars(self, parent_free_vars: Set[str]):
        # subtract vars bound by parent from free_vars
        for var in [*self._parent_eval_bindings,
                    *self._parent_agg_bindings,
                    *self._parent_scan_bindings]:
            self.free_vars.pop(var, 0)
        # subtract vars that will be bound by inserted lets
        for (var, _) in self.lifted_lets.values():
            self.free_vars.pop(var, 0)
        # update parent's free variables
        parent_free_vars.update(self.free_vars)


class CSERenderer(Renderer):
    def __init__(self, stop_at_jir=False):
        self.stop_at_jir = stop_at_jir
        self.jir_count = 0
        self.jirs = {}
        self.memo: Dict[int, Sequence[str]] = {}
        self.uid_count = 0
        self.binding_sites: Dict[int, BindingSite] = {}

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

    @staticmethod
    def find_in_scope(x: 'ir.BaseIR', context: List[StackFrame], outermost_scope: int) -> int:
        for i in range(len(context) - 1, outermost_scope - 1, -1):
            if id(x) in context[i].visited:
                return i
        return -1

    @staticmethod
    def lifted_in_scope(x: 'ir.BaseIR', context: List[StackFrame]) -> int:
        for i in range(len(context) - 1, -1, -1):
            if id(x) in context[i].lifted_lets:
                return i
        return -1

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

    class State:
        ...
        def __init__(self, x, children, builder, context, outermost_scope, depth):
            self.x = x
            self.children = children
            self.builder = builder
            self.context = context
            self.outermost_scope = outermost_scope
            self.ir_child_num = 0
            self.depth = depth
            self.i = 0

        def copy(self):
            new_state = CSERenderer.State(self.x, self.children, self.builder, self.context, self.outermost_scope, self.depth)
            new_state.ir_child_num = self.ir_child_num
            new_state.i = self.i
            return new_state

        def pre_add_lets(self, binding_sites):
            x = self.x
            insert_lets = id(x) in binding_sites and len(binding_sites[id(x)].lifted_lets) > 0
            if insert_lets:
                self.builder = []
                self.context.append(PrintStackFrame(binding_sites[id(x)]))
            head = x.render_head(self)
            if head != '':
                self.builder.append(head)
            return insert_lets

    def loop(self, state, k):
        if state.i >= len(state.children):
            return self.apply(k)
        state.builder.append(' ')
        child = state.children[state.i]

        new_state = self.State(child, child.render_children(self), state.builder, state.context, state.outermost_scope, state.depth)

        if isinstance(state.x, ir.BaseIR):
            if state.x.new_block(state.ir_child_num):
                new_state.outermost_scope = state.depth
        if isinstance(child, ir.BaseIR):
            new_state.depth += 1
            lift_to = self.lifted_in_scope(child, state.context)
        else:
            lift_to = -1

        if (lift_to >= 0 and
                state.context[lift_to] and
                state.context[lift_to].depth >= state.outermost_scope):
            state.ir_child_num += 1
            (name, _) = state.context[lift_to].lifted_lets[id(child)]

            if id(child) in state.context[lift_to].visited:
                state.builder.append(f'(Ref {name})')
                state.i += 1
                return self.loop(state, k)

            state.context[lift_to].visited[id(child)] = child

            new_state.builder = [f'(Let {name} ']

            if id(child) in self.memo:
                pcl = self.PostChildrenLifted(state, state.builder, new_state.builder, self.memo[id(child)], lift_to, name, k)
                return self.apply(pcl)

            insert_lets = new_state.pre_add_lets(self.binding_sites)
            assert(not insert_lets)

            pcl = self.PostChildrenLifted(state, state.builder, new_state.builder, [child.render_tail(self)], lift_to, name, k)

            return self.loop(new_state, pcl)

        if isinstance(child, ir.BaseIR):
            if id(child) in self.memo:
                new_state.builder.append(*self.memo[id(child)])
                state.i += 1
                return self.loop(state, k)

            insert_lets = new_state.pre_add_lets(self.binding_sites)

            pc = self.PostChildren(state, state.builder, new_state.builder, [child.render_tail(self)], insert_lets, k)

            return self.loop(new_state, pc)
        else:
            head = child.render_head(self)
            if head != '':
                state.builder.append(head)

            new_state.ir_child_num = state.ir_child_num
            insert_lets = False

            pc = self.PostChildren(state, state.builder, new_state.builder, new_state.x.render_tail(self), insert_lets, k)

            return self.loop(new_state, pc)

    class Kont:
        pass

    class PostChildrenLifted(Kont):
        def __init__(self, state, builder, local_builder, tail, lift_to, name, k):
            self.state = state
            self.builder = builder
            self.k = k
            self.local_builder = local_builder
            self.tail = tail
            self.lift_to = lift_to
            self.name = name

    class PostChildren(Kont):
        def __init__(self, state, builder, local_builder, tail, insert_lets, k):
            self.state = state
            self.builder = builder
            self.k = k
            self.local_builder = local_builder
            self.tail = tail
            self.insert_lets = insert_lets

    class PostRoot(Kont):
        def __init__(self, state, builder, local_builder, tail, insert_lets):
            self.state = state
            self.builder = builder
            self.local_builder = local_builder
            self.tail = tail
            self.insert_lets = insert_lets

    def apply(self, k: Kont):
        if isinstance(k, self.PostChildrenLifted):
            k.local_builder.extend(k.tail)
            k.local_builder.append(' ')
            # let_bodies is built post-order, which guarantees earlier
            # lets can't refer to later lets
            k.state.context[k.lift_to].let_bodies.append(k.local_builder)
            k.builder.append(f'(Ref {k.name})')
            k.state.i += 1
            return self.loop(k.state, k.k)
        elif isinstance(k, self.PostChildren):
            k.local_builder.extend(k.tail)
            if k.insert_lets:
                self.add_lets(k.state.context, k.local_builder, k.builder)
            if not isinstance(k.state.x, ir.BaseIR):
                k.state.ir_child_num += 1
            k.state.i += 1
            return self.loop(k.state, k.k)
        else:
            k.local_builder.append(k.tail)
            if k.insert_lets:
                self.add_lets(k.state.context, k.local_builder, k.builder)
            return ''.join(k.builder)

    @staticmethod
    def add_lets(context, local_builder, builder):
        sf = context[-1]
        context.pop()
        for let_body in sf.let_bodies:
            builder.extend(let_body)
        builder.extend(local_builder)
        num_lets = len(sf.lifted_lets)
        for _ in range(num_lets):
            builder.append(')')

    def compute_new_bindings(self, root: 'ir.BaseIR'):
        # force computation of all types
        root.typ

        root_frame = StackFrame(0, ({}, {}, {}), root)
        stack = [root_frame]
        binding_sites = {}

        while True:
            frame = stack[-1]
            node = frame.node
            child_idx = frame.child_idx
            frame.child_idx += 1

            if child_idx >= len(node.children):
                if len(stack) <= 1:
                    break

                parent_frame = stack[-2]

                # mark node as visited at potential let insertion site
                stack[frame.bind_depth()].visited[id(node)] = node

                # if any lets being inserted here, add node to registry of
                # binding sites
                if frame.lifted_lets:
                    binding_sites[id(node)] = \
                        BindingSite(frame.lifted_lets, len(stack))

                frame.update_parent_free_vars(parent_frame.free_vars)

                stack.pop()
                continue

            child = node.children[child_idx]

            if self.stop_at_jir and hasattr(child, '_jir'):
                self.memo[id(child)] = self.add_jir(child)
                continue

            seen_in_scope = self.find_in_scope(child, stack,
                                               frame.min_binding_depth)

            if seen_in_scope >= 0 and isinstance(child, ir.IR):
                # we've seen 'child' before, should not traverse (or we will
                # find too many lifts)
                if id(child) in stack[seen_in_scope].lifted_lets:
                    (uid, _) = stack[seen_in_scope].lifted_lets[id(child)]
                else:
                    # second time we've seen 'child', lift to a let
                    uid = self.uid()
                    stack[seen_in_scope].lifted_lets[id(child)] = (uid, child)
                # Since we are not traversing 'child', we don't know its free
                # variables. To prevent a parent from being lifted too high,
                # we must register 'child' as having the free variable 'uid',
                # which will be true when 'child' is replaced by "Ref uid".
                frame.free_vars[uid] = seen_in_scope
                continue

            # first time visiting 'child'

            if isinstance(child, ir.Ref):
                if child.name not in (var for (var, _) in
                                      node.bindings(child_idx)):
                    (eval_c, _, _) = node.child_context_without_bindings(
                        child_idx, frame.context)
                    frame.free_vars.update({child.name: eval_c[child.name]})
                continue

            stack.append(frame.make_child_frame(len(stack)))
            continue

        for (var, _) in root_frame.lifted_lets.values():
            var_depth = root_frame.free_vars.pop(var, 0)
            assert var_depth == 0
        assert(len(root_frame.free_vars) == 0)

        binding_sites[id(root)] = BindingSite(frame.lifted_lets, 0)
        return binding_sites

    def __call__(self, root: 'ir.BaseIR') -> str:
        self.binding_sites = self.compute_new_bindings(root)
        builder = []
        state = self.State(root, root.render_children(self), builder, [], 0, 1)
        if id(root) in self.memo:
            state.builder.append(self.memo[id(root)])
            return ''.join(builder)
        insert_lets = id(root) in self.binding_sites and len(self.binding_sites[id(root)].lifted_lets) > 0
        if insert_lets:
            state.builder = []
            state.context.append(PrintStackFrame(self.binding_sites[id(root)]))
        head = root.render_head(self)
        if head != '':
            state.builder.append(head)

        pc = self.PostRoot(state, builder, state.builder, root.render_tail(self), insert_lets)

        return self.loop(state, pc)
