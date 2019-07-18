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


class CSERenderer(object):
    def __init__(self, stop_at_jir=False):
        self.stop_at_jir = stop_at_jir
        self.jir_count = 0
        self.jirs = {}
        self.memo: Dict[int, Sequence[str]] = {}
        self.uid_count = 0

    def uid(self) -> str:
        self.uid_count += 1
        return f'__cse_{self.uid_count}'

    def add_jir(self, x):
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

    def block_local(self, x: 'BaseIR') -> Sequence[str]:
        visited: Set[int] = set()
        lift: Dict[int, str] = {}

        def find_lifts(x: 'BaseIR'):
            visited.add(id(x))
            for i in range(0, len(x.children)):
                child = x.children[i]
                if id(child) in visited:
                    # for now, only add non-relational lets
                    if id(child) not in lift and isinstance(child, ir.IR):
                        # second time we've seen 'x', lift to a let
                        lift[id(child)] = self.uid()
                elif self.stop_at_jir and hasattr(child, '_jir'):
                    self.memo[id(child)] = self.add_jir(child)
                elif x.new_block(i):
                    self.memo[id(child)] = self.block_local(child)
                    visited.add(id(child))
                else:
                    find_lifts(child)

        find_lifts(x)
        visited = set()
        # later lets may refer to earlier lets
        lets = []

        def add_lets(x: 'BaseIR', builder):
            visited.add(id(x))
            head = x.render_head(self)
            if head != '':
                builder.append(head)
            for i in range(0, len(x.children)):
                builder.append(' ')
                child = x.children[i]
                first_visit = id(child) not in visited
                new_block = id(child) in self.new_blocks
                if id(child) in lift:
                    name = lift[id(child)]
                    if first_visit:
                        local_builder = []
                        if new_block:
                            local_builder = self.new_blocks[id(child)]
                        else:
                            add_lets(child, local_builder)
                        lets.extend(f'(Let {name} ')
                        lets.extend(local_builder)
                        lets.extend(' ')
                    builder.extend(f'(Ref {name})')
                else:
                    if new_block:
                        builder.extend(self.new_blocks[id(child)])
                    else:
                        add_lets(child, builder)
            builder.append(x.render_tail(self))

        builder = []
        add_lets(x, builder)

        num_lets = len(lift)
        lets.extend(builder)
        for i in range(num_lets):
            lets.extend(')')

        return lets

    def __call__(self, x: 'BaseIR') -> str:
        return ''.join(self.block_local(x))
