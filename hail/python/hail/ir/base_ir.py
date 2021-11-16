import abc

from hail.utils.java import Env
from .renderer import Renderer, PlainRenderer, Renderable


counter = 0


def get_next_int():
    global counter
    counter = counter + 1
    return counter


def _env_bind(env, bindings):
    if bindings:
        if env:
            res = env.copy()
            res.update(bindings)
            return res
        else:
            return dict(bindings)
    else:
        return env


class BaseIR(Renderable):
    def __init__(self, *children):
        super().__init__()
        self._type = None
        self.children = children
        self._error_id = None
        self._stack_trace = None

    def __str__(self):
        r = PlainRenderer(stop_at_jir=False)
        return r(self)

    def render_head(self, r: Renderer):
        head_str = self.head_str()

        if head_str != '':
            head_str = f' {head_str}'
        trailing_space = ''
        if len(self.children) > 0:
            trailing_space = ' '
        return f'({self._ir_name()}{head_str}{trailing_space}'

    def render_tail(self, r: Renderer):
        return ')'

    def _ir_name(self):
        return self.__class__.__name__

    def render_children(self, r: Renderer):
        return self.children

    def head_str(self):
        """String to be added after IR name in serialized representation.

        Returns
        -------
        str
        """
        return ''

    @property
    @abc.abstractmethod
    def typ(self):
        return

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.children == other.children and self._eq(other)

    def __ne__(self, other):
        return not self == other

    def _eq(self, other):
        """Compare non-child-BaseIR attributes of the BaseIR.

        Parameters
        ----------
        other
            BaseIR of the same class.

        Returns
        -------
        bool
        """
        return True

    def __hash__(self):
        return 31 + hash(str(self))

    def new_block(self, i: int) -> bool:
        return self.renderable_new_block(self.renderable_idx_of_child(i))

    @abc.abstractmethod
    def renderable_new_block(self, i: int) -> bool:
        return self.new_block(i)

    @staticmethod
    def is_effectful() -> bool:
        return False

    def bindings(self, i: int, default_value=None):
        """Compute variables bound in child 'i'.

        Returns
        -------
        dict
            mapping from bound variables to 'default_value', if provided,
            otherwise to their types
        """
        return self.renderable_bindings(self.renderable_idx_of_child(i), default_value)

    def renderable_bindings(self, i: int, default_value=None):
        return {}

    def agg_bindings(self, i: int, default_value=None):
        return self.renderable_agg_bindings(self.renderable_idx_of_child(i), default_value)

    def renderable_agg_bindings(self, i: int, default_value=None):
        return {}

    def scan_bindings(self, i: int, default_value=None):
        return self.renderable_scan_bindings(self.renderable_idx_of_child(i), default_value)

    def renderable_scan_bindings(self, i: int, default_value=None):
        return {}

    def uses_agg_context(self, i: int) -> bool:
        return self.renderable_uses_agg_context(self.renderable_idx_of_child(i))

    def renderable_uses_agg_context(self, i: int) -> bool:
        return False

    def uses_scan_context(self, i: int) -> bool:
        return self.renderable_uses_scan_context(self.renderable_idx_of_child(i))

    def renderable_uses_scan_context(self, i: int) -> bool:
        return False

    def renderable_idx_of_child(self, i: int) -> int:
        return i

    # Used as a variable, bound by any node which defines the meaning of
    # aggregations (e.g. MatrixMapRows, AggFilter, etc.), and "referenced" by
    # any node which performs aggregations (e.g. AggFilter, ApplyAggOp, etc.).
    agg_capability = 'agg_capability'

    @classmethod
    def uses_agg_capability(cls) -> bool:
        return False

    def renderable_child_context_without_bindings(self, i: int, parent_context):
        (eval_c, agg_c, scan_c) = parent_context
        if self.renderable_uses_agg_context(i):
            return (agg_c, None, None)
        elif self.renderable_uses_scan_context(i):
            return (scan_c, None, None)
        else:
            return parent_context

    def child_context(self, i: int, parent_context, default_value=None):
        return self.renderable_child_context(self.renderable_idx_of_child(i), parent_context, default_value)

    def renderable_child_context(self, i: int, parent_context, default_value=None):
        base = self.renderable_child_context_without_bindings(i, parent_context)
        eval_b = self.bindings(i, default_value)
        agg_b = self.agg_bindings(i, default_value)
        scan_b = self.scan_bindings(i, default_value)
        if eval_b or agg_b or scan_b:
            (eval_c, agg_c, scan_c) = base
            return _env_bind(eval_c, eval_b), _env_bind(agg_c, agg_b), _env_bind(scan_c, scan_b)
        else:
            return base

    @property
    def free_vars(self):
        return set()

    @property
    def free_agg_vars(self):
        return set()

    @property
    def free_scan_vars(self):
        return set()

    def base_search(self, criteria):
        others = [node for child in self.children if isinstance(child, BaseIR) for node in child.base_search(criteria)]
        if criteria(self):
            return others + [self]
        return others

    def save_error_info(self):
        self._error_id = get_next_int()

        import traceback
        stack = traceback.format_stack()
        i = len(stack)
        while i > 0:
            candidate = stack[i - 1]
            if 'IPython' in candidate:
                break
            i -= 1

        forbidden_phrases = [
            '_ir_lambda_method',
            'decorator.py',
            'decorator-gen',
            'typecheck/check',
            'interactiveshell.py',
            'expressions.construct_variable',
            'traceback.format_stack()'
        ]
        filt_stack = [
            candidate for candidate in stack[i:]
            if not any(phrase in candidate for phrase in forbidden_phrases)
        ]

        self._stack_trace = '\n'.join(filt_stack)


class IR(BaseIR):
    def __init__(self, *children):
        super().__init__(*children)
        self._aggregations = None
        self._free_vars = None
        self._free_agg_vars = None
        self._free_scan_vars = None

    @property
    def aggregations(self):
        if self._aggregations is None:
            self._aggregations = [agg for child in self.children for agg in child.aggregations]
        return self._aggregations

    @property
    def is_nested_field(self):
        return False

    def search(self, criteria):
        others = [node for child in self.children if isinstance(child, IR) for node in child.search(criteria)]
        if criteria(self):
            return others + [self]
        return others

    def copy(self, *args):
        raise NotImplementedError("IR has no copy method defined.")

    def map_ir(self, f):
        new_children = []
        for child in self.children:
            if isinstance(child, IR):
                new_children.append(f(child))
            else:
                new_children.append(child)

        return self.copy(*new_children)

    @property
    def bound_variables(self):
        return {v for child in self.children if isinstance(child, IR) for v in child.bound_variables}

    @property
    def typ(self):
        if self._type is None:
            self._compute_type({}, None)
            assert self._type is not None, self
        return self._type

    def renderable_new_block(self, i: int) -> bool:
        return False

    @abc.abstractmethod
    def _compute_type(self, env, agg_env):
        raise NotImplementedError(self)

    @property
    def free_vars(self):
        def vars_from_child(i):
            if self.uses_agg_context(i):
                assert(len(self.children[i].free_agg_vars) == 0)
                return set()
            if self.uses_scan_context(i):
                assert(len(self.children[i].free_scan_vars) == 0)
                return set()
            return self.children[i].free_vars.difference(self.bindings(i, 0).keys())

        if self._free_vars is None:
            self._free_vars = {
                var for i in range(len(self.children))
                for var in vars_from_child(i)}
            if self.uses_agg_capability():
                self._free_vars.add(BaseIR.agg_capability)
        return self._free_vars

    @property
    def free_agg_vars(self):
        def vars_from_child(i):
            if self.uses_agg_context(i):
                return self.children[i].free_vars
            return self.children[i].free_agg_vars.difference(self.agg_bindings(i, 0).keys())

        if self._free_agg_vars is None:
            self._free_agg_vars = {
                var for i in range(len(self.children))
                for var in vars_from_child(i)}
        return self._free_agg_vars

    @property
    def free_scan_vars(self):
        def vars_from_child(i):
            if self.uses_scan_context(i):
                return self.children[i].free_vars
            return self.children[i].free_scan_vars.difference(self.scan_bindings(i, 0).keys())

        if self._free_scan_vars is None:
            self._free_scan_vars = {
                var for i in range(len(self.children))
                for var in vars_from_child(i)}
        return self._free_scan_vars


class TableIR(BaseIR):
    def __init__(self, *children):
        super().__init__(*children)

    @abc.abstractmethod
    def _compute_type(self):
        ...

    @property
    def typ(self):
        if self._type is None:
            self._compute_type()
            assert self._type is not None, self
        return self._type

    def renderable_new_block(self, i: int) -> bool:
        return True

    global_env = {'global'}
    row_env = {'global', 'row'}


class MatrixIR(BaseIR):
    def __init__(self, *children):
        super().__init__(*children)

    @abc.abstractmethod
    def _compute_type(self):
        ...

    @property
    def typ(self):
        if self._type is None:
            self._compute_type()
            assert self._type is not None, self
        return self._type

    def renderable_new_block(self, i: int) -> bool:
        return True

    global_env = {'global'}
    row_env = {'global', 'va'}
    col_env = {'global', 'sa'}
    entry_env = {'global', 'sa', 'va', 'g'}


class BlockMatrixIR(BaseIR):
    def __init__(self, *children):
        super().__init__(*children)

    @abc.abstractmethod
    def _compute_type(self):
        ...

    @property
    def typ(self):
        if self._type is None:
            self._compute_type()
            assert self._type is not None, self
        return self._type

    def renderable_new_block(self, i: int) -> bool:
        return True


class JIRVectorReference(object):
    def __init__(self, jid, length, item_type):
        self.jid = jid
        self.length = length
        self.item_type = item_type

    def __len__(self):
        return self.length

    def __del__(self):
        # can't do anything if the hail context is stopped
        if Env._hc is None:
            return
        try:
            Env.backend()._jhc.pyRemoveIrVector(self.jid)
        # there is only so much we can do if the attempt to remove the unused IR fails,
        # especially since this will often get called during interpreter shutdown.
        except Exception:
            pass
