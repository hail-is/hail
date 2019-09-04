from hail import ir
import abc
from typing import Sequence, List, Set, Dict, Callable, Tuple, Optional
from collections import namedtuple


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


Vars = Dict[str, int]
Context = (Vars, Vars, Vars)


class AnalysisStackFrame:
    __slots__ = ['min_binding_depth', 'min_value_binding_depth', 'scan_scope',
                 'context', 'node', '_free_vars', 'visited', 'agg_visited',
                 'scan_visited', 'lifted_lets', 'agg_lifted_lets',
                 'scan_lifted_lets', 'child_idx', '_child_bindings']

    def __init__(self, min_binding_depth: int, min_value_binding_depth: int,
                 scan_scope: bool, context: Context, x: 'ir.BaseIR'):
        # immutable
        self.min_binding_depth = min_binding_depth
        self.min_value_binding_depth = min_value_binding_depth
        self.scan_scope = scan_scope
        self.context = context
        self.node = x
        # mutable
        self._free_vars = {}
        self.visited: Set[int] = set()
        self.agg_visited: Set[int] = set()
        self.scan_visited: Set[int] = set()
        self.lifted_lets: Dict[int, str] = {}
        self.agg_lifted_lets: Dict[int, str] = {}
        self.scan_lifted_lets: Dict[int, str] = {}
        self.child_idx = 0
        self._child_bindings = None

    # compute depth at which we might bind this node
    def bind_depth(self) -> int:
        if len(self._free_vars) > 0:
            bind_depth = max(self._free_vars.values())
            bind_depth = max(bind_depth, self.min_binding_depth)
        else:
            bind_depth = self.min_binding_depth
        return bind_depth

    def make_child_frame(self, depth: int) -> 'AnalysisStackFrame':
        x = self.node
        i = self.child_idx - 1
        child = x.children[i]
        child_min_binding_depth = self.min_binding_depth
        child_min_value_binding_depth = self.min_value_binding_depth
        child_scan_scope = self.scan_scope
        if x.new_block(i):
            child_min_binding_depth = depth
            child_min_value_binding_depth = depth
        elif x.uses_agg_context(i):
            child_min_value_binding_depth = depth
            child_scan_scope = False
        elif x.uses_scan_context(i):
            child_min_value_binding_depth = depth
            child_scan_scope = True

        # compute vars bound in 'child' by 'node'
        eval_bindings = x.bindings(i, 0).keys()
        agg_bindings = x.agg_bindings(i, 0).keys()
        scan_bindings = x.scan_bindings(i, 0).keys()
        new_bindings = (eval_bindings, agg_bindings, scan_bindings)
        self._child_bindings = new_bindings
        child_context = x.child_context(i, self.context, depth)

        return AnalysisStackFrame(child_min_binding_depth, child_min_value_binding_depth, child_scan_scope, child_context, child,)

    def free_vars(self):
        # subtract vars that will be bound by inserted lets
        def bound_vars():
            yield from self.lifted_lets.values()
            yield from self.agg_lifted_lets.values()
            yield from self.scan_lifted_lets.values()
        for var in bound_vars():
            self._free_vars.pop(var, 0)
        return self._free_vars

    def update_free_vars(self, child_frame: 'AnalysisStackFrame'):
        child_free_vars = child_frame.free_vars()
        # subtract vars bound by parent from free_vars
        (eval_bindings, agg_bindings, scan_bindings) = self._child_bindings
        for var in [*eval_bindings, *agg_bindings, *scan_bindings]:
            child_free_vars.pop(var, 0)
        # update parent's free variables
        self._free_vars.update(child_free_vars)

    def update_parent_free_vars(self, parent_frame: 'AnalysisStackFrame'):
        # subtract vars bound by parent from free_vars
        (eval_bindings, agg_bindings, scan_bindings) = parent_frame._child_bindings
        for var in [*eval_bindings, *agg_bindings, *scan_bindings]:
            self._free_vars.pop(var, 0)
        # subtract vars that will be bound by inserted lets
        for (var, _) in self.lifted_lets.values():
            self._free_vars.pop(var, 0)
        # update parent's free variables
        parent_frame._free_vars.update(self._free_vars)


BindingSite = namedtuple(
    'BindingSite',
    'depth lifted_lets agg_lifted_lets scan_lifted_lets')
BindingsStackFrame = namedtuple(
    'BindingsStackFrame',
    'depth lifted_lets agg_lifted_lets scan_lifted_lets visited agg_visited'
    ' scan_visited let_bodies')

def make_bindings_stack_frame(site: BindingSite):
    return BindingsStackFrame(depth=site.depth,
                              lifted_lets=site.lifted_lets,
                              agg_lifted_lets=site.agg_lifted_lets,
                              scan_lifted_lets=site.scan_lifted_lets,
                              visited={},
                              agg_visited={},
                              scan_visited={},
                              let_bodies=[])


class PrintStackFrame:
    __slots__ = ['node', 'children', 'min_binding_depth',
                 'min_value_binding_depth', 'scan_scope', 'depth',
                 'lift_to_frame', 'insert_lets', 'builder', 'child_idx']

    def __init__(self, node, children, builder, min_binding_depth,
                 min_value_binding_depth, scan_scope, depth, insert_lets,
                 lift_to_frame=None):
        # immutable
        self.node: Renderable = node
        self.children: Sequence[Renderable] = children
        self.min_binding_depth: int = min_binding_depth
        self.min_value_binding_depth: int = min_value_binding_depth
        self.scan_scope: bool = scan_scope
        self.depth: int = depth
        self.lift_to_frame: Optional[BindingsStackFrame] = lift_to_frame
        self.insert_lets: bool = insert_lets
        # mutable
        self.builder = builder
        self.child_idx = -1

    def add_lets(self, let_bodies, out_builder):
        for let_body in let_bodies:
            out_builder.extend(let_body)
        out_builder.extend(self.builder)
        num_lets = len(let_bodies)
        for _ in range(num_lets):
            out_builder.append(')')

    def make_child_frame(self, renderer, binding_sites, builder, context,
                         min_binding_depth, min_value_binding_depth, scan_scope,
                         depth):
        child = self.children[self.child_idx]
        return self.make(child, renderer, binding_sites, builder, context,
                         min_binding_depth, min_value_binding_depth, scan_scope,
                         depth)

    @staticmethod
    def make(node, renderer, binding_sites, builder, context, min_binding_depth,
             min_value_binding_depth, scan_scope, depth):
        insert_lets = (id(node) in binding_sites
                       and (len(binding_sites[id(node)].lifted_lets) > 0
                            or len(binding_sites[id(node)].agg_lifted_lets) > 0))
        state = PrintStackFrame(node, node.render_children(renderer), builder,
                                min_binding_depth, min_value_binding_depth,
                                scan_scope, depth, insert_lets)
        if insert_lets:
            state.builder = []
            context.append(make_bindings_stack_frame(binding_sites[id(node)]))
        head = node.render_head(renderer)
        if head != '':
            state.builder.append(head)
        return state


class CSERenderer:
    def __init__(self, stop_at_jir=False):
        self.stop_at_jir = stop_at_jir
        self.jir_count = 0
        self.jirs = {}
        self.memo: Dict[int, Sequence[str]] = {}
        self.uid_count = 0

    def uid(self) -> str:
        self.uid_count += 1
        return f'__cse_{self.uid_count}'

    def add_jir(self, jir):
        jir_id = f'm{self.jir_count}'
        self.jir_count += 1
        self.jirs[jir_id] = jir
        return jir_id

    # At top of main loop, we are considering the node 'node' and its
    # 'child_idx'th child, or if 'child_idx' = 'len(node.children)', we are
    # about to do post-processing on 'node' before moving back up to its parent.
    #
    # 'stack' is a stack of 'AnalysisStackFrame's, one for each node on the path
    # from 'root' to 'node. Each stack frame tracks the following immutable
    # information:
    # * 'node': The node corresponding to this stack frame.
    # * 'min_binding_depth': If 'node' is to be bound in a let, the let may not
    #   rise higher than this (its depth must be >= 'min_binding_depth')
    # * 'context': The binding context of 'node'. Maps variables bound above
    #   to the depth at which they were bound (more precisely, if
    #   'context[var] == depth', then 'stack[depth-1].node' binds 'var' in the
    #   subtree rooted at 'stack[depth].node').
    # * 'new_bindings': The variables which were bound by 'node's parent. These
    #   must be subtracted out of 'node's free variables when updating the
    #   parent's free variables.
    #
    # Each stack frame also holds the following mutable state:
    # * 'free_vars': The running union of free variables in the subtree rooted
    #   at 'node'.
    # * 'visited': A set of visited descendants. For each descendant 'x' of
    #   'node', 'id(x)' is added to 'visited'. This allows us to recognize when
    #   we see a node for a second time.
    # * 'lifted_lets': For each descendant 'x' of 'node', if 'x' is to be bound
    #   in a let immediately above 'node', then 'lifted_lets' contains 'id(x)',
    #   along with the unique id to bind 'x' to.
    # * 'child_idx': the child currently being visited (satisfies the invariant
    #   'stack[i].node.children[stack[i].child_idx] is stack[i+1].node').
    #
    # Returns a Dict summarizing of all lets to be inserted. For each 'node'
    # which will have lets inserted immediately above it, maps 'id(node)' to a
    # 'BindingSite' recording the depth of 'node' and a Dict 'lifted_lets',
    # where for each descendant 'x' which will be bound above 'node',
    # 'lifted_lets' maps 'id(x)' to the unique id 'x' will be bound to.
    def compute_new_bindings(self, root: 'ir.BaseIR') -> Dict[int, BindingSite]:
        root_frame = AnalysisStackFrame(0, 0, False, ({}, {}, {}), root)
        stack = [root_frame]
        binding_sites = {}

        while True:
            frame = stack[-1]
            node = frame.node
            child_idx = frame.child_idx
            frame.child_idx += 1

            if child_idx >= len(node.children):
                # mark node as visited at potential let insertion site
                # FIXME: factor out to AnalysisStackFrame method
                if not node.is_effectful():
                    bind_depth = frame.bind_depth()
                    if bind_depth < frame.min_value_binding_depth:
                        if frame.scan_scope:
                            stack[bind_depth].scan_visited.add(id(node))
                        else:
                            stack[bind_depth].agg_visited.add(id(node))
                    else:
                        stack[bind_depth].visited.add(id(node))

                # if any lets being inserted here, add node to registry of
                # binding sites
                if frame.lifted_lets or frame.agg_lifted_lets or frame.scan_lifted_lets:
                    binding_sites[id(node)] = BindingSite(
                        lifted_lets=frame.lifted_lets,
                        agg_lifted_lets=frame.agg_lifted_lets,
                        scan_lifted_lets=frame.scan_lifted_lets,
                        depth=len(stack))

                stack.pop()
                if not stack:
                    break
                stack[-1].update_free_vars(frame)
                continue

            child = node.children[child_idx]

            if self.stop_at_jir and hasattr(child, '_jir'):
                jir_id = self.add_jir(child._jir)
                if isinstance(child, ir.MatrixIR):
                    jref = f'(JavaMatrix {jir_id})'
                elif isinstance(child, ir.TableIR):
                    jref = f'(JavaTable {jir_id})'
                elif isinstance(child, ir.BlockMatrixIR):
                    jref = f'(JavaBlockMatrix {jir_id})'
                else:
                    assert isinstance(child, ir.IR)
                    jref = f'(JavaIR {jir_id})'

                self.memo[id(child)] = jref
                continue

            if isinstance(child, ir.IR):
                seen_in_scope = next((i for i in reversed(range(frame.min_value_binding_depth,
                                                                len(stack)))
                                      if id(child) in stack[i].visited),
                                     None)
                lets = None
                if seen_in_scope is not None:
                    lets = stack[seen_in_scope].lifted_lets
                else:
                    if frame.scan_scope:
                        seen_in_scope = next(
                            (i for i in reversed(range(frame.min_binding_depth,
                                                       frame.min_value_binding_depth))
                             if id(child) in stack[i].scan_visited),
                            None)
                        if seen_in_scope is not None:
                            lets = stack[seen_in_scope].scan_lifted_lets
                    else:
                        seen_in_scope = next(
                            (i for i in reversed(range(frame.min_binding_depth,
                                                       frame.min_value_binding_depth))
                             if id(child) in stack[i].agg_visited),
                            None)
                        if seen_in_scope is not None:
                            lets = stack[seen_in_scope].agg_lifted_lets

                if lets is not None:
                    # we've seen 'child' before, should not traverse (or we will
                    # find too many lifts)
                    if id(child) in lets:
                        uid = lets[id(child)]
                    else:
                        # second time we've seen 'child', lift to a let
                        uid = self.uid()
                        lets[id(child)] = uid
                    # Since we are not traversing 'child', we don't know its free
                    # variables. To prevent a parent from being lifted too high,
                    # we must register 'child' as having the free variable 'uid',
                    # which will be true when 'child' is replaced by "(Ref uid)".
                    frame._free_vars[uid] = seen_in_scope
                    continue

            # first time visiting 'child'

            if isinstance(child, ir.Ref):
                if child.name not in node.bindings(child_idx, default_value=0).keys():
                    (eval_c, _, _) = node.child_context_without_bindings(
                        child_idx, frame.context)
                    try:
                        frame._free_vars[child.name] = eval_c[child.name]
                    except:
                        raise
                continue

            stack.append(frame.make_child_frame(len(stack)))
            continue

        root_free_vars = root_frame.free_vars()
        assert(len(root_free_vars) == 0)

        binding_sites[id(root)] = BindingSite(
            lifted_lets=frame.lifted_lets,
            agg_lifted_lets=frame.agg_lifted_lets,
            scan_lifted_lets=frame.scan_lifted_lets,
            depth=0)
        return binding_sites

    # At top of main loop, we are considering the 'Renderable' 'node' and its
    # 'child_idx'th child, or if 'child_idx' = 'len(node.children)', we are
    # about to do post-processing on 'node' before moving back up to its parent.
    #
    # 'stack' is a stack of 'PrintStackFrame's, one for each node on the path
    # from 'root' to 'node. Each stack frame tracks the following immutable
    # information:
    # * 'node': The 'Renderable' node corresponding to this stack frame.
    # * 'children': The list of 'Renderable' children.
    # * 'min_binding_depth': If 'node' is to be bound in a let, the let may not
    #   rise higher than this (its depth must be >= 'min_binding_depth')
    # * 'depth': The depth of 'node' in the original tree, i.e. the number of
    #   BaseIR in 'stack', not counting other 'Renderable's.
    # * 'lift_to_frame': The outermost frame in which 'node' was marked to be
    #   lifted in the analysis pass, if any, otherwise None.
    # * 'insert_lets': True if any lets need to be inserted above 'node'. No
    #   node has both 'lift_to_frame' not None and 'insert_lets' True.
    #
    # Each stack frame also holds the following mutable state:
    # * 'child_idx': The index of the 'Renderable' child currently being
    #   visited.
    # * 'builder': The buffer building the parent's rendered IR.
    # * 'local_builder': The builder building 'node's IR.
    # If 'insert_lets', all lets will be added to 'builder' before copying
    # 'local_builder' to 'builder. If 'lift_to_frame', 'local_builder' will be
    # added to 'lift_to_frame's list of lifted lets, while only "(Ref ...)" will
    # be added to 'builder'. If neither, then it is safe for 'local_builder' to
    # be 'builder', to save copying.

    def build_string(self, root, binding_sites):
        root_builder = []
        context: List[BindingsStackFrame] = []

        if id(root) in self.memo:
            return ''.join(self.memo[id(root)])

        stack = [PrintStackFrame.make(root, self, binding_sites, root_builder, context, 0, 0, False, 1)]

        while True:
            frame = stack[-1]
            node = frame.node
            frame.child_idx += 1
            child_idx = frame.child_idx
            if isinstance(node, ir.ApplyAggOp) and node.agg_op == 'Count':
                print('...')

            if child_idx >= len(frame.children):
                if frame.lift_to_frame is not None:
                    assert(not frame.insert_lets)
                    if id(node) in self.memo:
                        frame.builder.append(self.memo[id(node)])
                    else:
                        frame.builder.append(node.render_tail(self))
                    frame.builder.append(' ')
                    # let_bodies is built post-order, which guarantees earlier
                    # lets can't refer to later lets
                    frame.lift_to_frame.let_bodies.append(frame.builder)
                    stack.pop()
                    continue
                else:
                    frame.builder.append(node.render_tail(self))
                    stack.pop()
                    if frame.insert_lets:
                        if not stack:
                            out_builder = root_builder
                        else:
                            out_builder = stack[-1].builder
                        frame.add_lets(context[-1].let_bodies, out_builder)
                        context.pop()
                    if not stack:
                        return ''.join(root_builder)
                    continue

            if child_idx > 0:
                frame.builder.append(' ')
            child = frame.children[child_idx]

            child_min_binding_depth = frame.min_binding_depth
            child_min_value_binding_depth = frame.min_value_binding_depth
            child_scan_scope = False
            child_depth = frame.depth

            if isinstance(node, ir.BaseIR):
                if node.renderable_new_block(child_idx):
                    child_min_binding_depth = frame.depth
                    child_min_value_binding_depth = frame.depth
                if node.renderable_uses_agg_context(child_idx):
                    child_min_value_binding_depth = frame.depth
                    child_scan_scope = False
                if node.renderable_uses_scan_context(child_idx):
                    child_min_value_binding_depth = frame.depth
                    child_scan_scope = True

            if isinstance(child, ir.BaseIR):
                child_depth += 1
                lift_to_frame = next(
                    (c for c in context
                     if c.depth >= child_min_value_binding_depth
                     and id(child) in c.lifted_lets),
                    None)
                if lift_to_frame:
                    lift_type = 'value'
                else:
                    lift_to_frame = next(
                        (c for c in context
                         if child_min_binding_depth <= c.depth < child_min_value_binding_depth
                         and id(child) in c.agg_lifted_lets),
                        None)
                    if lift_to_frame:
                        lift_type = 'agg'
                    else:
                        lift_to_frame = next(
                            (c for c in context
                             if child_min_binding_depth <= c.depth < child_min_value_binding_depth
                             and id(child) in c.scan_lifted_lets),
                            None)
                        if lift_to_frame:
                            lift_type = 'scan'
            else:
                lift_to_frame = None

            if lift_to_frame:
                insert_lets = (id(child) in binding_sites
                               and (len(binding_sites[id(child)].lifted_lets) > 0
                                    or len(binding_sites[id(child)].agg_lifted_lets) > 0
                                    or len(binding_sites[id(child)].scan_lifted_lets > 0)))
                assert not insert_lets
                if lift_type == 'value':
                    name = lift_to_frame.lifted_lets[id(child)]
                    child_builder = [f'(Let {name} ']
                    visited = lift_to_frame.visited
                elif lift_type == 'agg':
                    name = lift_to_frame.agg_lifted_lets[id(child)]
                    child_builder = [f'(AggLet {name} False ']
                    visited = lift_to_frame.agg_visited
                else:
                    name = lift_to_frame.scan_lifted_lets[id(child)]
                    child_builder = [f'(AggLet {name} True ']
                    visited = lift_to_frame.scan_visited

                frame.builder.append(f'(Ref {name})')

                if id(child) in self.memo:
                    child_builder.append(self.memo[id(child)])
                    child_builder.append(' ')
                    lift_to_frame.let_bodies.append(child_builder)
                    continue
                    # new_state = PrintStackFrame(child, [], child_builder, child_min_binding_depth, child_depth, insert_lets, lift_to_frame)
                    # stack.append(new_state)
                    # continue

                if id(child) in visited:
                    continue
                visited[id(child)] = child

                new_state = frame.make_child_frame(self, binding_sites,
                                                   child_builder, context,
                                                   child_min_binding_depth,
                                                   child_min_value_binding_depth,
                                                   child_scan_scope,
                                                   child_depth)
                new_state.lift_to_frame = lift_to_frame
                stack.append(new_state)
                continue

            if id(child) in self.memo:
                frame.builder.append(self.memo[id(child)])
                continue

            new_state = frame.make_child_frame(self, binding_sites,
                                               frame.builder, context,
                                               child_min_binding_depth,
                                               child_min_value_binding_depth,
                                               child_scan_scope,
                                               child_depth)
            stack.append(new_state)
            continue

    def __call__(self, root: 'ir.BaseIR') -> str:
        binding_sites = self.compute_new_bindings(root)
        return self.build_string(root, binding_sites)
