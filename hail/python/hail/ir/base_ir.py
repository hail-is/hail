import abc

from hail.expr.types import tstream
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
        raise NotImplementedError

    @property
    def is_stream(self):
        return False

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

    def copy(self, *args):
        raise NotImplementedError("IR has no copy method defined.")

    def new_block(self, i: int) -> bool:
        return self.renderable_new_block(self.renderable_idx_of_child(i))

    @abc.abstractmethod
    def renderable_new_block(self, i: int) -> bool:
        return self.new_block(i)

    @staticmethod
    def is_effectful() -> bool:
        return False

    @property
    @abc.abstractmethod
    def uses_randomness(self) -> bool:
        pass

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
        self.has_uids = False
        self.needs_randomness_handling = False

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

    def map_ir(self, f):
        new_children = []
        for child in self.children:
            if isinstance(child, IR):
                new_children.append(f(child))
            else:
                new_children.append(child)

        return self.copy(*new_children)

    @property
    def uses_randomness(self) -> bool:
        return '__rng_state' in self.free_vars or '__rng_state' in self.free_agg_vars or '__rng_state' in self.free_scan_vars

    @property
    def uses_value_randomness(self):
        return '__rng_state' in self.free_vars

    def uses_agg_randomness(self, is_scan) -> bool:
        if is_scan:
            return '__rng_state' in self.free_scan_vars
        else:
            return '__rng_state' in self.free_agg_vars

    @property
    def bound_variables(self):
        return {v for child in self.children if isinstance(child, IR) for v in child.bound_variables}

    @property
    def typ(self):
        if self._type is None:
            self.compute_type({}, None, deep_typecheck=False)
        return self._type

    def renderable_new_block(self, i: int) -> bool:
        return False

    def compute_type(self, env, agg_env, deep_typecheck):
        if deep_typecheck or self._type is None:
            computed = self._compute_type(env, agg_env, deep_typecheck)
            assert(computed is not None)
            if self._type is not None:
                assert self._type == computed
            self._type = computed

    def assign_type(self, typ):
        if self._type is None:
            computed = self._compute_type({}, None, deep_typecheck=False)
            if computed is not None:
                assert computed == typ, (computed, typ)
            self._type = typ
        else:
            assert self._type == typ

    @abc.abstractmethod
    def _compute_type(self, env, agg_env, deep_typecheck):
        raise NotImplementedError(self)

    @abc.abstractmethod
    def _handle_randomness(self, create_uids):
        pass

    @property
    def might_be_stream(self):
        return type(self)._handle_randomness != IR._handle_randomness

    @property
    def is_stream(self):
        return self.might_be_stream and isinstance(self.typ, tstream)

    def handle_randomness(self, create_uids):
        """Elaborate rng semantics in stream typed IR.

        Recursive transformation of stream typed IRs. Ensures that all
        contained seeded randomness gets a unique rng state on every stream
        iteration. Optionally inserts a uid in the returned stream element type.
        The uid may be an int64, or arbitrary tuple of int64s. The only
        requirement is that all stream elements contain distinct uid values.
        """
        assert(self.is_stream)
        if (create_uids == self.has_uids) and not self.needs_randomness_handling:
            return self
        new = self._handle_randomness(create_uids)
        new.has_uids = create_uids
        new.needs_randomness_handling = False
        return new

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
                return self.children[i].free_vars.difference(self.bindings(i, 0).keys())
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
                return self.children[i].free_vars.difference(self.bindings(i, 0).keys())
            return self.children[i].free_scan_vars.difference(self.scan_bindings(i, 0).keys())

        if self._free_scan_vars is None:
            self._free_scan_vars = {
                var for i in range(len(self.children))
                for var in vars_from_child(i)}
        return self._free_scan_vars


class TableIR(BaseIR):
    def __init__(self, *children):
        super().__init__(*children)
        self._children_use_randomness = any(child.uses_randomness for child in children)

    @abc.abstractmethod
    def _compute_type(self, deep_typecheck):
        ...

    def compute_type(self, deep_typecheck):
        if deep_typecheck or self._type is None:
            computed = self._compute_type(deep_typecheck)
            if self._type is not None:
                assert self._type == computed
            else:
                self._type = computed

    @property
    def typ(self):
        if self._type is None:
            self.compute_type(deep_typecheck=False)
        return self._type

    @property
    def uses_randomness(self) -> bool:
        return self._children_use_randomness

    @abc.abstractmethod
    def _handle_randomness(self, uid_field_name):
        pass

    def handle_randomness(self, uid_field_name):
        """Elaborate rng semantics

        Recursively transform IR to ensure that all contained seeded randomness
        gets a unique rng state on every table row. Optionally inserts a uid
        field in the returned table. The uid may be an int64, or arbitrary
        tuple of int64s. The only requirement is that all table rows contain
        distinct uid values.
        """
        if uid_field_name is None and not self.uses_randomness:
            return self
        return self._handle_randomness(uid_field_name)

    def renderable_new_block(self, i: int) -> bool:
        return True

    global_env = {'global'}
    row_env = {'global', 'row'}


class MatrixIR(BaseIR):
    def __init__(self, *children):
        super().__init__(*children)
        self._children_use_randomness = any(child.uses_randomness for child in children)

    @property
    def uses_randomness(self) -> bool:
        return self._children_use_randomness

    def handle_randomness(self, row_uid_field_name, col_uid_field_name):
        """Elaborate rng semantics

        Recursively transform IR to ensure that all contained seeded randomness
        gets a unique rng state on every evaluation. Optionally inserts a uid
        row field and/or column field in the returned matrix table. The uids may
        be an int64, or arbitrary tuple of int64s. The only requirement is that
        all rows contain distinct uid values, and likewise for columns.
        """
        if row_uid_field_name is None and col_uid_field_name is None and not self.uses_randomness:
            return self
        result = self._handle_randomness(row_uid_field_name, col_uid_field_name)
        assert result is not None
        assert row_uid_field_name is None or row_uid_field_name in result.typ.row_type
        assert col_uid_field_name is None or col_uid_field_name in result.typ.col_type
        return result

    @abc.abstractmethod
    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        pass

    @abc.abstractmethod
    def _compute_type(self, deep_typecheck):
        ...

    def compute_type(self, deep_typecheck):
        if deep_typecheck or self._type is None:
            computed = self._compute_type(deep_typecheck)
            if self._type is not None:
                assert self._type == computed
            else:
                self._type = computed

    @property
    def typ(self):
        if self._type is None:
            self.compute_type(deep_typecheck=False)
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
        self._children_use_randomness = any(child.uses_randomness for child in children)

    @property
    def uses_randomness(self) -> bool:
        return self._children_use_randomness

    @abc.abstractmethod
    def _compute_type(self, deep_typecheck):
        ...

    def compute_type(self, deep_typecheck):
        if deep_typecheck or self._type is None:
            computed = self._compute_type(deep_typecheck)
            if self._type is not None:
                assert self._type == computed
            else:
                self._type = computed

    @property
    def typ(self):
        if self._type is None:
            self.compute_type(deep_typecheck=False)
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
