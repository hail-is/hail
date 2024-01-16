import abc
import builtins
import json
import math
import pprint
from collections.abc import Mapping, Sequence
from typing import ClassVar, Union

import numpy as np
import pandas as pd

import hail as hl
from hailtop.frozendict import frozendict
from hailtop.hail_frozenlist import frozenlist

from .. import genetics
from ..genetics.reference_genome import reference_genome_type
from ..typecheck import nullable, oneof, transformed, typecheck, typecheck_method
from ..utils.byte_reader import ByteReader, ByteWriter
from ..utils.java import escape_parsable
from ..utils.misc import lookup_bit
from ..utils.struct import Struct
from .nat import NatBase, NatLiteral
from .type_parsing import type_grammar, type_node_visitor

__all__ = [
    'dtype',
    'dtypes_from_pandas',
    'HailType',
    'hail_type',
    'is_container',
    'is_compound',
    'is_numeric',
    'is_primitive',
    'types_match',
    'tint',
    'tint32',
    'tint64',
    'tfloat',
    'tfloat32',
    'tfloat64',
    'tstr',
    'tbool',
    'tarray',
    'tstream',
    'tndarray',
    'tset',
    'tdict',
    'tstruct',
    'tunion',
    'ttuple',
    'tinterval',
    'tlocus',
    'tcall',
    'tvoid',
    'tvariable',
    'hts_entry_schema',
]


def summary_type(t):
    if isinstance(t, hl.tdict):
        return f'dict<{summary_type(t.key_type)}, {summary_type(t.value_type)}>'
    elif isinstance(t, hl.tset):
        return f'set<{summary_type(t.element_type)}>'
    elif isinstance(t, hl.tarray):
        return f'array<{summary_type(t.element_type)}>'
    elif isinstance(t, hl.tstruct):
        return f'struct with {len(t)} fields'
    elif isinstance(t, hl.ttuple):
        return f'tuple with {len(t)} fields'
    elif isinstance(t, hl.tinterval):
        return f'interval<{summary_type(t.point_type)}>'
    else:
        return str(t)


def dtype(type_str) -> 'HailType':
    r"""Parse a type from its string representation.

    Examples
    --------

    >>> hl.dtype('int')
    dtype('int32')

    >>> hl.dtype('float')
    dtype('float64')

    >>> hl.dtype('array<int32>')
    dtype('array<int32>')

    >>> hl.dtype('dict<str, bool>')
    dtype('dict<str, bool>')

    >>> hl.dtype('struct{a: int32, `field with spaces`: int64}')
    dtype('struct{a: int32, `field with spaces`: int64}')

    Notes
    -----
    This function is able to reverse ``str(t)`` on a :class:`.HailType`.

    The grammar is defined as follows:

    .. code-block:: text

        type = _ (array / set / dict / struct / union / tuple / interval / int64 / int32 / float32 / float64 / bool / str / call / str / locus) _
        int64 = "int64" / "tint64"
        int32 = "int32" / "tint32" / "int" / "tint"
        float32 = "float32" / "tfloat32"
        float64 = "float64" / "tfloat64" / "tfloat" / "float"
        bool = "tbool" / "bool"
        call = "tcall" / "call"
        str = "tstr" / "str"
        locus = ("tlocus" / "locus") _ "[" identifier "]"
        array = ("tarray" / "array") _ "<" type ">"
        array = ("tstream" / "stream") _ "<" type ">"
        ndarray = ("tndarray" / "ndarray") _ "<" type, identifier ">"
        set = ("tset" / "set") _ "<" type ">"
        dict = ("tdict" / "dict") _ "<" type "," type ">"
        struct = ("tstruct" / "struct") _ "{" (fields / _) "}"
        union = ("tunion" / "union") _ "{" (fields / _) "}"
        tuple = ("ttuple" / "tuple") _ "(" ((type ("," type)*) / _) ")"
        fields = field ("," field)*
        field = identifier ":" type
        interval = ("tinterval" / "interval") _ "<" type ">"
        identifier = _ (simple_identifier / escaped_identifier) _
        simple_identifier = ~"\w+"
        escaped_identifier = ~"`([^`\\\\]|\\\\.)*`"
        _ = ~"\s*"

    Parameters
    ----------
    type_str : :class:`str`
        String representation of type.

    Returns
    -------
    :class:`.HailType`
    """
    tree = type_grammar.parse(type_str)
    return type_node_visitor.visit(tree)


class HailTypeContext(object):
    def __init__(self, references=set()):
        self.references = references

    @property
    def is_empty(self):
        return len(self.references) == 0

    def _to_json_context(self):
        if self._json is None:
            self._json = {'reference_genomes': {r: hl.get_reference(r)._config for r in self.references}}
        return self._json

    @classmethod
    def union(cls, *types):
        ctxs = [t.get_context() for t in types if not t.get_context().is_empty]
        if len(ctxs) == 0:
            return _empty_context
        if len(ctxs) == 1:
            return ctxs[0]
        refs = ctxs[0].references.union(*[ctx.references for ctx in ctxs[1:]])
        return HailTypeContext(refs)


_empty_context = HailTypeContext()


class HailType(object):
    """
    Hail type superclass.
    """

    def __init__(self):
        super(HailType, self).__init__()
        self._context = None

    def __repr__(self):
        s = str(self).replace("'", "\\'")
        return "dtype('{}')".format(s)

    @abc.abstractmethod
    def _eq(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, HailType) and self._eq(other)

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __hash__(self):
        # FIXME this is a bit weird
        return 43 + hash(str(self))

    def pretty(self, indent=0, increment=4):
        """Returns a prettily formatted string representation of the type.

        Parameters
        ----------
        indent : :obj:`int`
            Spaces to indent.

        Returns
        -------
        :class:`str`
        """
        b = []
        b.append(' ' * indent)
        self._pretty(b, indent, increment)
        return ''.join(b)

    def _pretty(self, b, indent, increment):
        b.append(str(self))

    @abc.abstractmethod
    def _parsable_string(self) -> str:
        raise NotImplementedError

    def typecheck(self, value):
        """Check that `value` matches a type.

        Parameters
        ----------
        value
            Value to check.

        Raises
        ------
        :obj:`TypeError`
        """

        def check(t, obj):
            t._typecheck_one_level(obj)
            return True

        self._traverse(value, check)

    @abc.abstractmethod
    def _typecheck_one_level(self, annotation):
        raise NotImplementedError

    def _to_json(self, x):
        converted = self._convert_to_json_na(x)
        return json.dumps(converted)

    def _convert_to_json_na(self, x):
        if x is None:
            return x
        else:
            return self._convert_to_json(x)

    def _convert_to_json(self, x):
        return x

    def _from_json(self, s):
        x = json.loads(s)
        return self._convert_from_json_na(x)

    def _convert_from_json_na(self, x, _should_freeze: bool = False):
        if x is None:
            return x
        else:
            return self._convert_from_json(x, _should_freeze)

    def _convert_from_json(self, x, _should_freeze: bool = False):
        return x

    def _from_encoding(self, encoding):
        return self._convert_from_encoding(ByteReader(memoryview(encoding)))

    def _to_encoding(self, value) -> bytes:
        buf = bytearray()
        self._convert_to_encoding(ByteWriter(buf), value)
        return bytes(buf)

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False):
        raise ValueError("Not implemented yet")

    def _convert_to_encoding(self, byte_writer, value):
        raise ValueError("Not implemented yet")

    @staticmethod
    def _missing(value):
        return value is None or value is pd.NA

    def _traverse(self, obj, f):
        """Traverse a nested type and object.

        Parameters
        ----------
        obj : Any
        f : Callable[[HailType, Any], bool]
            Function to evaluate on the type and object. Traverse children if
            the function returns ``True``.
        """
        f(self, obj)

    @abc.abstractmethod
    def unify(self, t):
        raise NotImplementedError

    @abc.abstractmethod
    def subst(self):
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError

    def _get_context(self):
        return _empty_context

    def get_context(self):
        if self._context is None:
            self._context = self._get_context()
        return self._context

    def to_numpy(self):
        return object


hail_type = oneof(HailType, transformed((str, dtype)), type(None))


class _tvoid(HailType):
    def __init__(self):
        super(_tvoid, self).__init__()

    def __str__(self):
        return "void"

    def _eq(self, other):
        return isinstance(other, _tvoid)

    def _parsable_string(self):
        return "Void"

    def unify(self, t):
        return t == tvoid

    def subst(self):
        return self

    def clear(self):
        pass

    def _convert_from_encoding(self, *_):
        raise ValueError("Cannot decode void type")

    def _convert_to_encoding(self, *_):
        raise ValueError("Cannot encode void type")


class _tint32(HailType):
    """Hail type for signed 32-bit integers.

    Their values can range from :math:`-2^{31}` to :math:`2^{31} - 1`
    (approximately 2.15 billion).

    In Python, these are represented as :obj:`int`.
    """

    def __init__(self):
        super(_tint32, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation is not None:
            if not is_int32(annotation):
                raise TypeError("type 'tint32' expected Python 'int', but found type '%s'" % type(annotation))
            elif not self.min_value <= annotation <= self.max_value:
                raise TypeError(
                    f"Value out of range for 32-bit integer: "
                    f"expected [{self.min_value}, {self.max_value}], found {annotation}"
                )

    def __str__(self):
        return "int32"

    def _eq(self, other):
        return isinstance(other, _tint32)

    def _parsable_string(self):
        return "Int32"

    @property
    def min_value(self):
        return -(1 << 31)

    @property
    def max_value(self):
        return (1 << 31) - 1

    def unify(self, t):
        return t == tint32

    def subst(self):
        return self

    def clear(self):
        pass

    def to_numpy(self):
        return np.int32

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> int:
        return byte_reader.read_int32()

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        byte_writer.write_int32(value)

    def _byte_size(self):
        return 4


class _tint64(HailType):
    """Hail type for signed 64-bit integers.

    Their values can range from :math:`-2^{63}` to :math:`2^{63} - 1`.

    In Python, these are represented as :obj:`int`.
    """

    def __init__(self):
        super(_tint64, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation is not None:
            if not is_int64(annotation):
                raise TypeError("type 'int64' expected Python 'int', but found type '%s'" % type(annotation))
            if not self.min_value <= annotation <= self.max_value:
                raise TypeError(
                    f"Value out of range for 64-bit integer: "
                    f"expected [{self.min_value}, {self.max_value}], found {annotation}"
                )

    def __str__(self):
        return "int64"

    def _eq(self, other):
        return isinstance(other, _tint64)

    def _parsable_string(self):
        return "Int64"

    @property
    def min_value(self):
        return -(1 << 63)

    @property
    def max_value(self):
        return (1 << 63) - 1

    def unify(self, t):
        return t == tint64

    def subst(self):
        return self

    def clear(self):
        pass

    def to_numpy(self):
        return np.int64

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> int:
        return byte_reader.read_int64()

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        byte_writer.write_int64(value)

    def _byte_size(self):
        return 8


class _tfloat32(HailType):
    """Hail type for 32-bit floating point numbers.

    In Python, these are represented as :obj:`float`.
    """

    def __init__(self):
        super(_tfloat32, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation is not None and not is_float32(annotation):
            raise TypeError("type 'float32' expected Python 'float', but found type '%s'" % type(annotation))

    def __str__(self):
        return "float32"

    def _eq(self, other):
        return isinstance(other, _tfloat32)

    def _parsable_string(self):
        return "Float32"

    def _convert_from_json(self, x, _should_freeze: bool = False):
        return float(x)

    def _convert_to_json(self, x):
        if math.isfinite(x):
            return x
        else:
            return str(x)

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> float:
        return byte_reader.read_float32()

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        byte_writer.write_float32(value)

    def unify(self, t):
        return t == tfloat32

    def subst(self):
        return self

    def clear(self):
        pass

    def to_numpy(self):
        return np.float32

    def _byte_size(self):
        return 4


class _tfloat64(HailType):
    """Hail type for 64-bit floating point numbers.

    In Python, these are represented as :obj:`float`.
    """

    def __init__(self):
        super(_tfloat64, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation is not None and not is_float64(annotation):
            raise TypeError("type 'float64' expected Python 'float', but found type '%s'" % type(annotation))

    def __str__(self):
        return "float64"

    def _eq(self, other):
        return isinstance(other, _tfloat64)

    def _parsable_string(self):
        return "Float64"

    def _convert_from_json(self, x, _should_freeze: bool = False):
        return float(x)

    def _convert_to_json(self, x):
        if math.isfinite(x):
            return x
        else:
            return str(x)

    def unify(self, t):
        return t == tfloat64

    def subst(self):
        return self

    def clear(self):
        pass

    def to_numpy(self):
        return np.float64

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> float:
        return byte_reader.read_float64()

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        byte_writer.write_float64(value)

    def _byte_size(self):
        return 8


class _tstr(HailType):
    """Hail type for text strings.

    In Python, these are represented as strings.
    """

    def __init__(self):
        super(_tstr, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation and not isinstance(annotation, str):
            raise TypeError("type 'str' expected Python 'str', but found type '%s'" % type(annotation))

    def __str__(self):
        return "str"

    def _eq(self, other):
        return isinstance(other, _tstr)

    def _parsable_string(self):
        return "String"

    def unify(self, t):
        return t == tstr

    def subst(self):
        return self

    def clear(self):
        pass

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> str:
        length = byte_reader.read_int32()
        str_literal = byte_reader.read_bytes(length).decode('utf-8')

        return str_literal

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        value_bytes = value.encode('utf-8')
        byte_writer.write_int32(len(value_bytes))
        byte_writer.write_bytes(value_bytes)


class _tbool(HailType):
    """Hail type for Boolean (``True`` or ``False``) values.

    In Python, these are represented as :obj:`bool`.
    """

    def __init__(self):
        super(_tbool, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation is not None and not isinstance(annotation, bool):
            raise TypeError("type 'bool' expected Python 'bool', but found type '%s'" % type(annotation))

    def __str__(self):
        return "bool"

    def _eq(self, other):
        return isinstance(other, _tbool)

    def _parsable_string(self):
        return "Boolean"

    def unify(self, t):
        return t == tbool

    def subst(self):
        return self

    def clear(self):
        pass

    def to_numpy(self):
        return bool

    def _byte_size(self):
        return 1

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> bool:
        return byte_reader.read_bool()

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        byte_writer.write_bool(value)


class _trngstate(HailType):
    def __init__(self):
        super(_trngstate, self).__init__()

    def __str__(self):
        return "rng_state"

    def _eq(self, other):
        return isinstance(other, _trngstate)

    def _parsable_string(self):
        return "RNGState"

    def unify(self, t):
        return t == trngstate

    def subst(self):
        return self

    def clear(self):
        pass


class tndarray(HailType):
    """Hail type for n-dimensional arrays.

    .. include:: _templates/experimental.rst

    In Python, these are represented as NumPy :obj:`numpy.ndarray`.

    Notes
    -----

    NDArrays contain elements of only one type, which is parameterized by
    `element_type`.

    Parameters
    ----------
    element_type : :class:`.HailType`
        Element type of array.
    ndim : int32
        Number of dimensions.

    See Also
    --------
    :class:`.NDArrayExpression`, :obj:`.nd.array`
    """

    @typecheck_method(element_type=hail_type, ndim=oneof(NatBase, int))
    def __init__(self, element_type, ndim):
        self._element_type = element_type
        self._ndim = NatLiteral(ndim) if isinstance(ndim, int) else ndim
        super(tndarray, self).__init__()

    @property
    def element_type(self):
        """NDArray element type.

        Returns
        -------
        :class:`.HailType`
            Element type.
        """
        return self._element_type

    @property
    def ndim(self):
        """NDArray number of dimensions.

        Returns
        -------
        :obj:`int`
            Number of dimensions.
        """
        assert isinstance(self._ndim, NatLiteral), "tndarray must be realized with a concrete number of dimensions"
        return self._ndim.n

    def _traverse(self, obj, f):
        if f(self, obj):
            for elt in np.nditer(obj, ['zerosize_ok']):
                self.element_type._traverse(elt.item(), f)

    def _typecheck_one_level(self, annotation):
        if annotation is not None and not isinstance(annotation, np.ndarray):
            raise TypeError("type 'ndarray' expected Python 'numpy.ndarray', but found type '%s'" % type(annotation))

    def __str__(self):
        return "ndarray<{}, {}>".format(self.element_type, self.ndim)

    def _eq(self, other):
        return isinstance(other, tndarray) and self.element_type == other.element_type and self.ndim == other.ndim

    def _pretty(self, b, indent, increment):
        b.append('ndarray<')
        self._element_type._pretty(b, indent, increment)
        b.append(', ')
        b.append(str(self.ndim))
        b.append('>')

    def _parsable_string(self):
        return f'NDArray[{self._element_type._parsable_string()},{self.ndim}]'

    def _convert_from_json(self, x, _should_freeze: bool = False) -> np.ndarray:
        if is_numeric(self._element_type):
            np_type = self.element_type.to_numpy()
            return np.ndarray(shape=x['shape'], buffer=np.array(x['data'], dtype=np_type), dtype=np_type)
        else:
            raise TypeError("Hail cannot currently return ndarrays of non-numeric or boolean type.")

    def _convert_to_json(self, x):
        data = x.flatten("C").tolist()

        strides = []
        axis_one_step_byte_size = x.itemsize
        for dimension_size in x.shape:
            strides.append(axis_one_step_byte_size)
            axis_one_step_byte_size *= dimension_size if dimension_size > 0 else 1

        json_dict = {"shape": x.shape, "data": data}
        return json_dict

    def clear(self):
        self._element_type.clear()
        self._ndim.clear()

    def unify(self, t):
        return isinstance(t, tndarray) and self._element_type.unify(t._element_type) and self._ndim.unify(t._ndim)

    def subst(self):
        return tndarray(self._element_type.subst(), self._ndim.subst())

    def _get_context(self):
        return self.element_type.get_context()

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> np.ndarray:
        shape = [byte_reader.read_int64() for i in range(self.ndim)]
        total_num_elements = np.product(shape, dtype=np.int64)

        if self.element_type in _numeric_types:
            element_byte_size = self.element_type._byte_size
            bytes_to_read = element_byte_size * total_num_elements
            buffer = byte_reader.read_bytes_view(bytes_to_read)
            return np.frombuffer(buffer, self.element_type.to_numpy, count=total_num_elements).reshape(shape)
        else:
            elements = [
                self.element_type._convert_from_encoding(byte_reader, _should_freeze) for i in range(total_num_elements)
            ]
            np_type = self.element_type.to_numpy()
            return np.ndarray(shape=shape, buffer=np.array(elements, dtype=np_type), dtype=np_type, order="F")

    def _convert_to_encoding(self, byte_writer, value: np.ndarray):
        for dim in value.shape:
            byte_writer.write_int64(dim)

        if value.size > 0:
            if self.element_type in _numeric_types:
                byte_writer.write_bytes(value.data)
            else:
                for elem in np.nditer(value, order='F'):
                    self.element_type._convert_to_encoding(byte_writer, elem)


class tarray(HailType):
    """Hail type for variable-length arrays of elements.

    In Python, these are represented as :obj:`list`.

    Notes
    -----
    Arrays contain elements of only one type, which is parameterized by
    `element_type`.

    Parameters
    ----------
    element_type : :class:`.HailType`
        Element type of array.

    See Also
    --------
    :class:`.ArrayExpression`, :class:`.CollectionExpression`,
    :func:`~hail.expr.functions.array`, :ref:`sec-collection-functions`
    """

    @typecheck_method(element_type=hail_type)
    def __init__(self, element_type):
        self._element_type = element_type
        super(tarray, self).__init__()

    @property
    def element_type(self):
        """Array element type.

        Returns
        -------
        :class:`.HailType`
            Element type.
        """
        return self._element_type

    def _traverse(self, obj, f):
        if f(self, obj):
            for elt in obj:
                self.element_type._traverse(elt, f)

    def _typecheck_one_level(self, annotation):
        if annotation is not None:
            if not isinstance(annotation, Sequence):
                raise TypeError("type 'array' expected Python 'list', but found type '%s'" % type(annotation))

    def __str__(self):
        return "array<{}>".format(self.element_type)

    def _eq(self, other):
        return isinstance(other, tarray) and self.element_type == other.element_type

    def _pretty(self, b, indent, increment):
        b.append('array<')
        self.element_type._pretty(b, indent, increment)
        b.append('>')

    def _parsable_string(self):
        return "Array[" + self.element_type._parsable_string() + "]"

    def _convert_from_json(self, x, _should_freeze: bool = False) -> Union[list, frozenlist]:
        ls = [self.element_type._convert_from_json_na(elt, _should_freeze) for elt in x]
        if _should_freeze:
            return frozenlist(ls)
        return ls

    def _convert_to_json(self, x):
        return [self.element_type._convert_to_json_na(elt) for elt in x]

    def _propagate_jtypes(self, jtype):
        self._element_type._add_jtype(jtype.elementType())

    def unify(self, t):
        return isinstance(t, tarray) and self.element_type.unify(t.element_type)

    def subst(self):
        return tarray(self.element_type.subst())

    def clear(self):
        self.element_type.clear()

    def _get_context(self):
        return self.element_type.get_context()

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> Union[list, frozenlist]:
        length = byte_reader.read_int32()

        num_missing_bytes = math.ceil(length / 8)
        missing_bytes = byte_reader.read_bytes_view(num_missing_bytes)

        decoded = []
        i = 0
        current_missing_byte = None
        while i < length:
            which_missing_bit = i % 8
            if which_missing_bit == 0:
                current_missing_byte = missing_bytes[i // 8]

            if lookup_bit(current_missing_byte, which_missing_bit):
                decoded.append(None)
            else:
                element_decoded = self.element_type._convert_from_encoding(byte_reader, _should_freeze)
                decoded.append(element_decoded)
            i += 1
        if _should_freeze:
            return frozenlist(decoded)
        return decoded

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        length = len(value)
        byte_writer.write_int32(length)
        i = 0
        while i < length:
            missing_byte = 0
            for j in range(min(8, length - i)):
                if HailType._missing(value[i + j]):
                    missing_byte |= 1 << j
            byte_writer.write_byte(missing_byte)
            i += 8

        for element in value:
            if not HailType._missing(element):
                self.element_type._convert_to_encoding(byte_writer, element)


class tstream(HailType):
    @typecheck_method(element_type=hail_type)
    def __init__(self, element_type):
        self._element_type = element_type
        super(tstream, self).__init__()

    @property
    def element_type(self):
        return self._element_type

    def _traverse(self, obj, f):
        if f(self, obj):
            for elt in obj:
                self.element_type._traverse(elt, f)

    def _typecheck_one_level(self, annotation):
        raise TypeError("type 'stream' is not realizable in Python")

    def __str__(self):
        return "stream<{}>".format(self.element_type)

    def _eq(self, other):
        return isinstance(other, tstream) and self.element_type == other.element_type

    def _pretty(self, b, indent, increment):
        b.append('stream<')
        self.element_type._pretty(b, indent, increment)
        b.append('>')

    def _parsable_string(self):
        return "Stream[" + self.element_type._parsable_string() + "]"

    def _convert_from_json(self, x, _should_freeze: bool = False) -> Union[list, frozenlist]:
        ls = [self.element_type._convert_from_json_na(elt, _should_freeze) for elt in x]
        if _should_freeze:
            return frozenlist(ls)
        return ls

    def _convert_to_json(self, x):
        return [self.element_type._convert_to_json_na(elt) for elt in x]

    def _propagate_jtypes(self, jtype):
        self._element_type._add_jtype(jtype.elementType())

    def unify(self, t):
        return isinstance(t, tstream) and self.element_type.unify(t.element_type)

    def subst(self):
        return tstream(self.element_type.subst())

    def clear(self):
        self.element_type.clear()

    def _get_context(self):
        return self.element_type.get_context()


def is_setlike(maybe_setlike):
    return isinstance(maybe_setlike, (frozenset, set))


class tset(HailType):
    """Hail type for collections of distinct elements.

    In Python, these are represented as :obj:`set`.

    Notes
    -----
    Sets contain elements of only one type, which is parameterized by
    `element_type`.

    Parameters
    ----------
    element_type : :class:`.HailType`
        Element type of set.

    See Also
    --------
    :class:`.SetExpression`, :class:`.CollectionExpression`,
    :func:`.set`, :ref:`sec-collection-functions`
    """

    @typecheck_method(element_type=hail_type)
    def __init__(self, element_type):
        self._element_type = element_type
        self._array_repr = tarray(element_type)
        super(tset, self).__init__()

    @property
    def element_type(self):
        """Set element type.

        Returns
        -------
        :class:`.HailType`
            Element type.
        """
        return self._element_type

    def _traverse(self, obj, f):
        if f(self, obj):
            for elt in obj:
                self.element_type._traverse(elt, f)

    def _typecheck_one_level(self, annotation):
        if annotation is not None:
            if not is_setlike(annotation):
                raise TypeError("type 'set' expected Python 'set', but found type '%s'" % type(annotation))

    def __str__(self):
        return "set<{}>".format(self.element_type)

    def _eq(self, other):
        return isinstance(other, tset) and self.element_type == other.element_type

    def _pretty(self, b, indent, increment):
        b.append('set<')
        self.element_type._pretty(b, indent, increment)
        b.append('>')

    def _parsable_string(self):
        return "Set[" + self.element_type._parsable_string() + "]"

    def _convert_from_json(self, x, _should_freeze: bool = False) -> Union[set, frozenset]:
        s = {self.element_type._convert_from_json_na(elt, _should_freeze=True) for elt in x}
        if _should_freeze:
            return frozenset(s)
        return s

    def _convert_to_json(self, x):
        return [self.element_type._convert_to_json_na(elt) for elt in x]

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> Union[set, frozenset]:
        s = self._array_repr._convert_from_encoding(byte_reader, _should_freeze=True)
        if _should_freeze:
            return frozenset(s)
        return set(s)

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        self._array_repr._convert_to_encoding(byte_writer, list(value))

    def _propagate_jtypes(self, jtype):
        self._element_type._add_jtype(jtype.elementType())

    def unify(self, t):
        return isinstance(t, tset) and self.element_type.unify(t.element_type)

    def subst(self):
        return tset(self.element_type.subst())

    def clear(self):
        self.element_type.clear()

    def _get_context(self):
        return self.element_type.get_context()


class _freeze_this_type(HailType):
    def __init__(self, t):
        self.t = t

    def _convert_from_json_na(self, x, _should_freeze: bool = False):
        return self.t._convert_from_json_na(x, _should_freeze=True)

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False):
        return self.t._convert_from_encoding(byte_reader, _should_freeze=True)

    def _convert_to_encoding(self, byte_writer, x):
        return self.t._convert_to_encoding(byte_writer, x)


class tdict(HailType):
    """Hail type for key-value maps.

    In Python, these are represented as :obj:`dict`.

    Notes
    -----
    Dicts parameterize the type of both their keys and values with
    `key_type` and `value_type`.

    Parameters
    ----------
    key_type: :class:`.HailType`
        Key type.
    value_type: :class:`.HailType`
        Value type.

    See Also
    --------
    :class:`.DictExpression`, :func:`.dict`, :ref:`sec-collection-functions`
    """

    @typecheck_method(key_type=hail_type, value_type=hail_type)
    def __init__(self, key_type, value_type):
        self._key_type = key_type
        self._value_type = value_type
        self._array_repr = tarray(tstruct(key=_freeze_this_type(key_type), value=value_type))
        super(tdict, self).__init__()

    @property
    def key_type(self):
        """Dict key type.

        Returns
        -------
        :class:`.HailType`
            Key type.
        """
        return self._key_type

    @property
    def value_type(self):
        """Dict value type.

        Returns
        -------
        :class:`.HailType`
            Value type.
        """
        return self._value_type

    @property
    def element_type(self):
        return tstruct(key=self._key_type, value=self._value_type)

    def _traverse(self, obj, f):
        if f(self, obj):
            for k, v in obj.items():
                self.key_type._traverse(k, f)
                self.value_type._traverse(v, f)

    def _typecheck_one_level(self, annotation):
        if annotation is not None:
            if not isinstance(annotation, Mapping):
                raise TypeError("type 'dict' expected Python 'Mapping', but found type '%s'" % type(annotation))

    def __str__(self):
        return "dict<{}, {}>".format(self.key_type, self.value_type)

    def _eq(self, other):
        return isinstance(other, tdict) and self.key_type == other.key_type and self.value_type == other.value_type

    def _pretty(self, b, indent, increment):
        b.append('dict<')
        self.key_type._pretty(b, indent, increment)
        b.append(', ')
        self.value_type._pretty(b, indent, increment)
        b.append('>')

    def _parsable_string(self):
        return "Dict[{},{}]".format(self.key_type._parsable_string(), self.value_type._parsable_string())

    def _convert_from_json(self, x, _should_freeze: bool = False) -> Union[dict, frozendict]:
        d = {
            self.key_type._convert_from_json_na(elt['key'], _should_freeze=True): self.value_type._convert_from_json_na(
                elt['value'], _should_freeze=_should_freeze
            )
            for elt in x
        }
        if _should_freeze:
            return frozendict(d)
        return d

    def _convert_to_json(self, x):
        return [
            {'key': self.key_type._convert_to_json(k), 'value': self.value_type._convert_to_json(v)}
            for k, v in x.items()
        ]

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> Union[dict, frozendict]:
        # NB: We ensure the key is always frozen with a wrapper on the key_type in the _array_repr.
        d = {}
        length = byte_reader.read_int32()
        for _ in range(length):
            element = self._array_repr.element_type._convert_from_encoding(byte_reader, _should_freeze)
            d[element.key] = element.value

        if _should_freeze:
            return frozendict(d)
        return d

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        length = len(value)
        byte_writer.write_int32(length)
        for k, v in value.items():
            self._array_repr.element_type._convert_to_encoding(byte_writer, {'key': k, 'value': v})

    def _propagate_jtypes(self, jtype):
        self._key_type._add_jtype(jtype.keyType())
        self._value_type._add_jtype(jtype.valueType())

    def unify(self, t):
        return isinstance(t, tdict) and self.key_type.unify(t.key_type) and self.value_type.unify(t.value_type)

    def subst(self):
        return tdict(self._key_type.subst(), self._value_type.subst())

    def clear(self):
        self.key_type.clear()
        self.value_type.clear()

    def _get_context(self):
        return HailTypeContext.union(self.key_type, self.value_type)


class tstruct(HailType, Mapping):
    """Hail type for structured groups of heterogeneous fields.

    In Python, these are represented as :class:`.Struct`.

    Hail's :class:`.tstruct` type is commonly used to compose types together to form nested
    structures. Structs can contain any combination of types, and are ordered mappings
    from field name to field type. Each field name must be unique.

    Structs are very common in Hail. Each component of a :class:`.Table` and :class:`.MatrixTable`
    is a struct:

    - :meth:`.Table.row`
    - :meth:`.Table.globals`
    - :meth:`.MatrixTable.row`
    - :meth:`.MatrixTable.col`
    - :meth:`.MatrixTable.entry`
    - :meth:`.MatrixTable.globals`

    Structs appear below the top-level component types as well. Consider the following join:

    >>> new_table = table1.annotate(table2_fields = table2.index(table1.key))

    This snippet adds a field to ``table1`` called ``table2_fields``. In the new table,
    ``table2_fields`` will be a struct containing all the non-key fields from ``table2``.

    Parameters
    ----------
    field_types : keyword args of :class:`.HailType`
        Fields.

    See Also
    --------
    :class:`.StructExpression`, :class:`.Struct`
    """

    @typecheck_method(field_types=hail_type)
    def __init__(*args, **field_types):
        if len(args) < 1:
            raise TypeError("__init__() missing 1 required positional argument: 'self'")
        if len(args) > 1:
            raise TypeError(f"__init__() takes 1 positional argument but {len(args)} were given")
        self = args[0]
        self._field_types = field_types
        self._fields = tuple(field_types)
        super(tstruct, self).__init__()

    @property
    def types(self):
        """Struct field types.

        Returns
        -------
        :obj:`tuple` of :class:`.HailType`
        """
        return tuple(self._field_types.values())

    @property
    def fields(self):
        """Struct field names.

        Returns
        -------
        :obj:`tuple` of :class:`str`
            Tuple of struct field names.
        """
        return self._fields

    def _traverse(self, obj, f):
        if f(self, obj):
            for k, v in obj.items():
                t = self[k]
                t._traverse(v, f)

    def _typecheck_one_level(self, annotation):
        if annotation:
            if isinstance(annotation, Mapping):
                s = set(self)
                for f in annotation:
                    if f not in s:
                        raise TypeError(
                            "type '%s' expected fields '%s', but found fields '%s'"
                            % (self, list(self), list(annotation))
                        )
            else:
                raise TypeError(
                    "type 'struct' expected type Mapping (e.g. dict or hail.utils.Struct), but found '%s'"
                    % type(annotation)
                )

    @typecheck_method(item=oneof(int, str))
    def __getitem__(self, item):
        if not isinstance(item, str):
            item = self._fields[item]
        return self._field_types[item]

    def __iter__(self):
        return iter(self._field_types)

    def __len__(self):
        return len(self._fields)

    def __str__(self):
        return "struct{{{}}}".format(', '.join('{}: {}'.format(escape_parsable(f), str(t)) for f, t in self.items()))

    def items(self):
        return self._field_types.items()

    def _eq(self, other):
        return (
            isinstance(other, tstruct)
            and self._fields == other._fields
            and all(self[f] == other[f] for f in self._fields)
        )

    def _pretty(self, b, indent, increment):
        if not self._fields:
            b.append('struct {}')
            return

        pre_indent = indent
        indent += increment
        b.append('struct {')
        for i, (f, t) in enumerate(self.items()):
            if i > 0:
                b.append(', ')
            b.append('\n')
            b.append(' ' * indent)
            b.append('{}: '.format(escape_parsable(f)))
            t._pretty(b, indent, increment)
        b.append('\n')
        b.append(' ' * pre_indent)
        b.append('}')

    def _parsable_string(self):
        return "Struct{{{}}}".format(
            ','.join('{}:{}'.format(escape_parsable(f), t._parsable_string()) for f, t in self.items())
        )

    def _convert_from_json(self, x, _should_freeze: bool = False) -> Struct:
        return Struct(**{f: t._convert_from_json_na(x.get(f), _should_freeze) for f, t in self._field_types.items()})

    def _convert_to_json(self, x):
        return {f: t._convert_to_json_na(x[f]) for f, t in self.items()}

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> Struct:
        num_missing_bytes = math.ceil(len(self) / 8)
        missing_bytes = byte_reader.read_bytes_view(num_missing_bytes)

        kwargs = {}

        current_missing_byte = None
        for i, (f, t) in enumerate(self._field_types.items()):
            which_missing_bit = i % 8
            if which_missing_bit == 0:
                current_missing_byte = missing_bytes[i // 8]

            if lookup_bit(current_missing_byte, which_missing_bit):
                kwargs[f] = None
            else:
                field_decoded = t._convert_from_encoding(byte_reader, _should_freeze)
                kwargs[f] = field_decoded

        return Struct(**kwargs)

    def _convert_to_encoding(self, byte_writer: ByteWriter, value):
        keys = list(self.keys())
        length = len(keys)
        i = 0
        while i < length:
            missing_byte = 0
            for j in range(min(8, length - i)):
                if HailType._missing(value[keys[i + j]]):
                    missing_byte |= 1 << j
            byte_writer.write_byte(missing_byte)
            i += 8

        for f, t in self.items():
            if not HailType._missing(value[f]):
                t._convert_to_encoding(byte_writer, value[f])

    def _is_prefix_of(self, other):
        return (
            isinstance(other, tstruct)
            and len(self._fields) <= len(other._fields)
            and all(x == y for x, y in zip(self._field_types.values(), other._field_types.values()))
        )

    def _concat(self, other):
        new_field_types = {}
        new_field_types.update(self._field_types)
        new_field_types.update(other._field_types)
        return tstruct(**new_field_types)

    def _insert(self, path, t):
        if not path:
            return t

        key = path[0]
        keyt = self.get(key)
        if not (keyt and isinstance(keyt, tstruct)):
            keyt = tstruct()
        return self._insert_fields(**{key: keyt._insert(path[1:], t)})

    def _insert_field(self, field, typ):
        return self._insert_fields(**{field: typ})

    def _insert_fields(self, **new_fields):
        new_field_types = {}
        new_field_types.update(self._field_types)
        new_field_types.update(new_fields)
        return tstruct(**new_field_types)

    def _drop_fields(self, fields):
        return tstruct(**{f: t for f, t in self.items() if f not in fields})

    def _select_fields(self, fields):
        return tstruct(**{f: self[f] for f in fields})

    def _index_path(self, path):
        t = self
        for p in path:
            t = t[p]
        return t

    def _rename(self, map):
        seen = {}
        new_field_types = {}

        for f0, t in self.items():
            f = map.get(f0, f0)
            if f in seen:
                raise ValueError(
                    "Cannot rename two fields to the same name: attempted to rename {} and {} both to {}".format(
                        repr(seen[f]), repr(f0), repr(f)
                    )
                )
            else:
                seen[f] = f0
                new_field_types[f] = t

        return tstruct(**new_field_types)

    def unify(self, t):
        if not (isinstance(t, tstruct) and len(self) == len(t)):
            return False
        for (f1, t1), (f2, t2) in zip(self.items(), t.items()):
            if not (f1 == f2 and t1.unify(t2)):
                return False
        return True

    def subst(self):
        return tstruct(**{f: t.subst() for f, t in self.items()})

    def clear(self):
        for f, t in self.items():
            t.clear()

    def _get_context(self):
        return HailTypeContext.union(*self.values())


class tunion(HailType, Mapping):
    @typecheck_method(case_types=hail_type)
    def __init__(self, **case_types):
        """Tagged union type.  Values of type union represent one of several
        heterogenous, named cases.

        Parameters
        ----------
        cases : keyword args of :class:`.HailType`
            The union cases.

        """

        super(tunion, self).__init__()
        self._case_types = case_types
        self._cases = tuple(case_types)

    @property
    def cases(self):
        """Return union case names.

        Returns
        -------
        :obj:`tuple` of :class:`str`
            Tuple of union case names
        """
        return self._cases

    @typecheck_method(item=oneof(int, str))
    def __getitem__(self, item):
        if isinstance(item, int):
            item = self._cases[item]
        return self._case_types[item]

    def __iter__(self):
        return iter(self._case_types)

    def __len__(self):
        return len(self._cases)

    def __str__(self):
        return "union{{{}}}".format(', '.join('{}: {}'.format(escape_parsable(f), str(t)) for f, t in self.items()))

    def _eq(self, other):
        return (
            isinstance(other, tunion) and self._cases == other._cases and all(self[c] == other[c] for c in self._cases)
        )

    def _pretty(self, b, indent, increment):
        if not self._cases:
            b.append('union {}')
            return

        pre_indent = indent
        indent += increment
        b.append('union {')
        for i, (f, t) in enumerate(self.items()):
            if i > 0:
                b.append(', ')
            b.append('\n')
            b.append(' ' * indent)
            b.append('{}: '.format(escape_parsable(f)))
            t._pretty(b, indent, increment)
        b.append('\n')
        b.append(' ' * pre_indent)
        b.append('}')

    def _parsable_string(self):
        return "Union{{{}}}".format(
            ','.join('{}:{}'.format(escape_parsable(f), t._parsable_string()) for f, t in self.items())
        )

    def unify(self, t):
        if not (isinstance(t, tunion) and len(self) == len(t)):
            return False
        for (f1, t1), (f2, t2) in zip(self.items(), t.items()):
            if not (f1 == f2 and t1.unify(t2)):
                return False
        return True

    def subst(self):
        return tunion(**{f: t.subst() for f, t in self.items()})

    def clear(self):
        for f, t in self.items():
            t.clear()

    def _get_context(self):
        return HailTypeContext.union(*self.values())


class ttuple(HailType, Sequence):
    """Hail type for tuples.

    In Python, these are represented as :obj:`tuple`.

    Parameters
    ----------
    types: varargs of :class:`.HailType`
        Element types.

    See Also
    --------
    :class:`.TupleExpression`
    """

    @typecheck_method(types=hail_type)
    def __init__(self, *types):
        self._types = types
        super(ttuple, self).__init__()

    @property
    def types(self):
        """Tuple element types.

        Returns
        -------
        :obj:`tuple` of :class:`.HailType`
        """
        return self._types

    def _traverse(self, obj, f):
        if f(self, obj):
            for t, elt in zip(self.types, obj):
                t._traverse(elt, f)

    def _typecheck_one_level(self, annotation):
        if annotation:
            if not isinstance(annotation, tuple):
                raise TypeError("type 'tuple' expected Python tuple, but found '%s'" % type(annotation))
            if len(annotation) != len(self.types):
                raise TypeError("%s expected tuple of size '%i', but found '%s'" % (self, len(self.types), annotation))

    @typecheck_method(item=int)
    def __getitem__(self, item):
        return self._types[item]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self._types)

    def __str__(self):
        return "tuple({})".format(", ".join([str(t) for t in self.types]))

    def _eq(self, other):
        from operator import eq

        return (
            isinstance(other, ttuple) and len(self.types) == len(other.types) and all(map(eq, self.types, other.types))
        )

    def _pretty(self, b, indent, increment):
        pre_indent = indent
        indent += increment
        b.append('tuple (')
        for i, t in enumerate(self.types):
            if i > 0:
                b.append(', ')
            b.append('\n')
            b.append(' ' * indent)
            t._pretty(b, indent, increment)
        b.append('\n')
        b.append(' ' * pre_indent)
        b.append(')')

    def _parsable_string(self):
        return "Tuple[{}]".format(",".join([t._parsable_string() for t in self.types]))

    def _convert_from_json(self, x, _should_freeze: bool = False) -> tuple:
        return tuple(self.types[i]._convert_from_json_na(x[i], _should_freeze) for i in range(len(self.types)))

    def _convert_to_json(self, x):
        return [self.types[i]._convert_to_json_na(x[i]) for i in range(len(self.types))]

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> tuple:
        num_missing_bytes = math.ceil(len(self) / 8)
        missing_bytes = byte_reader.read_bytes_view(num_missing_bytes)

        answer = []
        current_missing_byte = None
        for i, t in enumerate(self.types):
            which_missing_bit = i % 8
            if which_missing_bit == 0:
                current_missing_byte = missing_bytes[i // 8]

            if lookup_bit(current_missing_byte, which_missing_bit):
                answer.append(None)
            else:
                field_decoded = t._convert_from_encoding(byte_reader, _should_freeze)
                answer.append(field_decoded)

        return tuple(answer)

    def _convert_to_encoding(self, byte_writer, value):
        length = len(self)
        i = 0
        while i < length:
            missing_byte = 0
            for j in range(min(8, length - i)):
                if HailType._missing(value[i + j]):
                    missing_byte |= 1 << j
            byte_writer.write_byte(missing_byte)
            i += 8
        for i, t in enumerate(self.types):
            if not HailType._missing(value[i]):
                t._convert_to_encoding(byte_writer, value[i])

    def unify(self, t):
        if not (isinstance(t, ttuple) and len(self.types) == len(t.types)):
            return False
        for t1, t2 in zip(self.types, t.types):
            if not t1.unify(t2):
                return False
        return True

    def subst(self):
        return ttuple(*[t.subst() for t in self.types])

    def clear(self):
        for t in self.types:
            t.clear()

    def _get_context(self):
        return HailTypeContext.union(*self.types)


def allele_pair(j: int, k: int):
    assert j >= 0 and j <= 0xFFFF
    assert k >= 0 and k <= 0xFFFF
    return j | (k << 16)


def allele_pair_sqrt(i):
    k = int(math.sqrt(8 * float(i) + 1) / 2 - 0.5)
    assert k * (k + 1) // 2 <= i
    j = i - k * (k + 1) // 2
    # TODO another assert
    return allele_pair(j, k)


small_allele_pair = [
    allele_pair(0, 0),
    allele_pair(0, 1),
    allele_pair(1, 1),
    allele_pair(0, 2),
    allele_pair(1, 2),
    allele_pair(2, 2),
    allele_pair(0, 3),
    allele_pair(1, 3),
    allele_pair(2, 3),
    allele_pair(3, 3),
    allele_pair(0, 4),
    allele_pair(1, 4),
    allele_pair(2, 4),
    allele_pair(3, 4),
    allele_pair(4, 4),
    allele_pair(0, 5),
    allele_pair(1, 5),
    allele_pair(2, 5),
    allele_pair(3, 5),
    allele_pair(4, 5),
    allele_pair(5, 5),
    allele_pair(0, 6),
    allele_pair(1, 6),
    allele_pair(2, 6),
    allele_pair(3, 6),
    allele_pair(4, 6),
    allele_pair(5, 6),
    allele_pair(6, 6),
    allele_pair(0, 7),
    allele_pair(1, 7),
    allele_pair(2, 7),
    allele_pair(3, 7),
    allele_pair(4, 7),
    allele_pair(5, 7),
    allele_pair(6, 7),
    allele_pair(7, 7),
]


class _tcall(HailType):
    """Hail type for a diploid genotype.

    In Python, these are represented by :class:`.Call`.
    """

    def __init__(self):
        super(_tcall, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation is not None and not isinstance(annotation, genetics.Call):
            raise TypeError("type 'call' expected Python hail.genetics.Call, but found %s'" % type(annotation))

    def __str__(self):
        return "call"

    def _eq(self, other):
        return isinstance(other, _tcall)

    def _parsable_string(self):
        return "Call"

    def _convert_from_json(self, x, _should_freeze: bool = False) -> genetics.Call:
        if x == '-':
            return genetics.Call([])
        if x == '|-':
            return genetics.Call([], phased=True)
        if x[0] == '|':
            return genetics.Call([int(x[1:])], phased=True)

        n = len(x)
        i = 0
        while i < n:
            c = x[i]
            if c in '|/':
                break
            i += 1

        if i == n:
            return genetics.Call([int(x)])

        return genetics.Call([int(x[0:i]), int(x[i + 1 :])], phased=(c == '|'))

    def _convert_to_json(self, x):
        return str(x)

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> genetics.Call:
        int_rep = byte_reader.read_int32()

        ploidy = (int_rep >> 1) & 0x3
        phased = (int_rep & 1) == 1

        def allele_repr(c):
            return c >> 3

        def ap_j(p):
            return p & 0xFFFF

        def ap_k(p):
            return (p >> 16) & 0xFFFF

        def gt_allele_pair(i):
            if i < len(small_allele_pair):
                return small_allele_pair[i]
            else:
                return allele_pair_sqrt(i)

        def call_allele_pair(i):
            if phased:
                rep = allele_repr(i)
                p = gt_allele_pair(rep)
                j = ap_j(p)
                k = ap_k(p)
                return allele_pair(j, k - j)
            else:
                rep = allele_repr(i)
                return gt_allele_pair(rep)

        if ploidy == 0:
            alleles = []
        elif ploidy == 1:
            alleles = [allele_repr(int_rep)]
        elif ploidy == 2:
            p = call_allele_pair(int_rep)
            alleles = [ap_j(p), ap_k(p)]
        else:
            raise ValueError("Unsupported Ploidy")

        return genetics.Call(alleles, phased)

    def _convert_to_encoding(self, byte_writer, value: genetics.Call):
        int_rep = 0

        int_rep |= value.ploidy << 1
        if value.phased:
            int_rep |= 1

        def diploid_gt_index(j: int, k: int):
            assert j <= k
            return k * (k + 1) // 2 + j

        def allele_pair_rep(c: genetics.Call):
            [j, k] = c.alleles
            if c.phased:
                return diploid_gt_index(j, j + k)
            return diploid_gt_index(j, k)

        assert value.ploidy <= 2
        if value.ploidy == 1:
            int_rep |= value.alleles[0] << 3
        elif value.ploidy == 2:
            int_rep |= allele_pair_rep(value) << 3

        byte_writer.write_int32(int_rep)

    def unify(self, t):
        return t == tcall

    def subst(self):
        return self

    def clear(self):
        pass


class tlocus(HailType):
    """Hail type for a genomic coordinate with a contig and a position.

    In Python, these are represented by :class:`.Locus`.

    Parameters
    ----------
    reference_genome: :class:`.ReferenceGenome` or :class:`str`
        Reference genome to use.

    See Also
    --------
    :class:`.LocusExpression`, :func:`.locus`, :func:`.parse_locus`,
    :class:`.Locus`
    """

    struct_repr = tstruct(contig=_tstr(), pos=_tint32())

    @classmethod
    @typecheck_method(reference_genome=nullable(reference_genome_type))
    def _schema_from_rg(cls, reference_genome='default'):
        # must match TLocus.schemaFromRG
        if reference_genome is None:
            return hl.tstruct(contig=hl.tstr, position=hl.tint32)
        return cls(reference_genome)

    @typecheck_method(reference_genome=reference_genome_type)
    def __init__(self, reference_genome='default'):
        self._rg = reference_genome
        super(tlocus, self).__init__()

    def _typecheck_one_level(self, annotation):
        if annotation is not None:
            if not isinstance(annotation, genetics.Locus):
                raise TypeError(
                    "type '{}' expected Python hail.genetics.Locus, but found '{}'".format(self, type(annotation))
                )
            if not self.reference_genome == annotation.reference_genome:
                raise TypeError(
                    "type '{}' encountered Locus with reference genome {}".format(
                        self, repr(annotation.reference_genome)
                    )
                )

    def __str__(self):
        return "locus<{}>".format(escape_parsable(str(self.reference_genome)))

    def _parsable_string(self):
        return "Locus({})".format(escape_parsable(str(self.reference_genome)))

    def _eq(self, other):
        return isinstance(other, tlocus) and self.reference_genome == other.reference_genome

    @property
    def reference_genome(self):
        """Reference genome.

        Returns
        -------
        :class:`.ReferenceGenome`
            Reference genome.
        """
        if self._rg is None:
            self._rg = hl.default_reference()
        return self._rg

    def _pretty(self, b, indent, increment):
        b.append('locus<{}>'.format(escape_parsable(self.reference_genome.name)))

    def _convert_from_json(self, x, _should_freeze: bool = False) -> genetics.Locus:
        return genetics.Locus(x['contig'], x['position'], reference_genome=self.reference_genome)

    def _convert_to_json(self, x):
        return {'contig': x.contig, 'position': x.position}

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False) -> genetics.Locus:
        as_struct = tlocus.struct_repr._convert_from_encoding(byte_reader)
        return genetics.Locus(as_struct.contig, as_struct.pos, self.reference_genome)

    def _convert_to_encoding(self, byte_writer, value: genetics.Locus):
        tlocus.struct_repr._convert_to_encoding(byte_writer, {'contig': value.contig, 'pos': value.position})

    def unify(self, t):
        return isinstance(t, tlocus) and self.reference_genome == t.reference_genome

    def subst(self):
        return self

    def clear(self):
        pass

    def _get_context(self):
        return HailTypeContext(references={self.reference_genome.name})


class tinterval(HailType):
    """Hail type for intervals of ordered values.

    In Python, these are represented by :class:`.Interval`.

    Parameters
    ----------
    point_type: :class:`.HailType`
        Interval point type.

    See Also
    --------
    :class:`.IntervalExpression`, :class:`.Interval`, :func:`.interval`,
    :func:`.parse_locus_interval`
    """

    @typecheck_method(point_type=hail_type)
    def __init__(self, point_type):
        self._point_type = point_type
        self._struct_repr = tstruct(start=point_type, end=point_type, includes_start=hl.tbool, includes_end=hl.tbool)
        super(tinterval, self).__init__()

    @property
    def point_type(self):
        """Interval point type.

        Returns
        -------
        :class:`.HailType`
            Interval point type.
        """
        return self._point_type

    def _traverse(self, obj, f):
        if f(self, obj):
            self.point_type._traverse(obj.start, f)
            self.point_type._traverse(obj.end, f)

    def _typecheck_one_level(self, annotation):
        from hail.utils import Interval

        if annotation is not None:
            if not isinstance(annotation, Interval):
                raise TypeError(
                    "type '{}' expected Python hail.utils.Interval, but found {}".format(self, type(annotation))
                )
            if annotation.point_type != self.point_type:
                raise TypeError(
                    "type '{}' encountered Interval with point type {}".format(self, repr(annotation.point_type))
                )

    def __str__(self):
        return "interval<{}>".format(str(self.point_type))

    def _eq(self, other):
        return isinstance(other, tinterval) and self.point_type == other.point_type

    def _pretty(self, b, indent, increment):
        b.append('interval<')
        self.point_type._pretty(b, indent, increment)
        b.append('>')

    def _parsable_string(self):
        return "Interval[{}]".format(self.point_type._parsable_string())

    def _convert_from_json(self, x, _should_freeze: bool = False):
        from hail.utils import Interval

        return Interval(
            self.point_type._convert_from_json_na(x['start'], _should_freeze),
            self.point_type._convert_from_json_na(x['end'], _should_freeze),
            x['includeStart'],
            x['includeEnd'],
            point_type=self.point_type,
        )

    def _convert_to_json(self, x):
        return {
            'start': self.point_type._convert_to_json_na(x.start),
            'end': self.point_type._convert_to_json_na(x.end),
            'includeStart': x.includes_start,
            'includeEnd': x.includes_end,
        }

    def _convert_from_encoding(self, byte_reader, _should_freeze: bool = False):
        interval_as_struct = self._struct_repr._convert_from_encoding(byte_reader, _should_freeze)
        return hl.Interval(
            interval_as_struct.start,
            interval_as_struct.end,
            interval_as_struct.includes_start,
            interval_as_struct.includes_end,
            point_type=self.point_type,
        )

    def _convert_to_encoding(self, byte_writer, value):
        interval_dict = {
            'start': value.start,
            'end': value.end,
            'includes_start': value.includes_start,
            'includes_end': value.includes_end,
        }
        self._struct_repr._convert_to_encoding(byte_writer, interval_dict)

    def unify(self, t):
        return isinstance(t, tinterval) and self.point_type.unify(t.point_type)

    def subst(self):
        return tinterval(self.point_type.subst())

    def clear(self):
        self.point_type.clear()

    def _get_context(self):
        return self.point_type.get_context()


class Box(object):
    named_boxes: ClassVar = {}

    @staticmethod
    def from_name(name):
        if name in Box.named_boxes:
            return Box.named_boxes[name]
        b = Box()
        Box.named_boxes[name] = b
        return b

    def __init__(self):
        pass

    def unify(self, v):
        if hasattr(self, 'value'):
            return self.value == v
        self.value = v
        return True

    def clear(self):
        if hasattr(self, 'value'):
            del self.value

    def get(self):
        assert hasattr(self, 'value')
        return self.value


tvoid = _tvoid()


tint32 = _tint32()
"""Hail type for signed 32-bit integers.

Their values can range from :math:`-2^{31}` to :math:`2^{31} - 1`
(approximately 2.15 billion).

In Python, these are represented as :obj:`int`.

See Also
--------
:class:`.Int32Expression`, :func:`.int`, :func:`.int32`
"""


tint64 = _tint64()
"""Hail type for signed 64-bit integers.

Their values can range from :math:`-2^{63}` to :math:`2^{63} - 1`.

In Python, these are represented as :obj:`int`.

See Also
--------
:class:`.Int64Expression`, :func:`.int64`
"""

tint = tint32
"""Alias for :py:data:`.tint32`."""

tfloat32 = _tfloat32()
"""Hail type for 32-bit floating point numbers.

In Python, these are represented as :obj:`float`.

See Also
--------
:class:`.Float32Expression`, :func:`.float64`
"""

tfloat64 = _tfloat64()
"""Hail type for 64-bit floating point numbers.

In Python, these are represented as :obj:`float`.

See Also
--------
:class:`.Float64Expression`, :func:`.float`, :func:`.float64`
"""

tfloat = tfloat64
"""Alias for :py:data:`.tfloat64`."""

tstr = _tstr()
"""Hail type for text strings.

In Python, these are represented as strings.

See Also
--------
:class:`.StringExpression`, :func:`.str`
"""

tbool = _tbool()
"""Hail type for Boolean (``True`` or ``False``) values.

In Python, these are represented as :obj:`bool`.

See Also
--------
:class:`.BooleanExpression`, :func:`.bool`
"""

trngstate = _trngstate()

tcall = _tcall()
"""Hail type for a diploid genotype.

In Python, these are represented by :class:`.Call`.

See Also
--------
:class:`.CallExpression`, :class:`.Call`, :func:`.call`, :func:`.parse_call`,
:func:`.unphased_diploid_gt_index_call`
"""

hts_entry_schema = tstruct(GT=tcall, AD=tarray(tint32), DP=tint32, GQ=tint32, PL=tarray(tint32))

_numeric_types = {_tbool, _tint32, _tint64, _tfloat32, _tfloat64}
_primitive_types = _numeric_types.union({_tstr})
_interned_types = _primitive_types.union({_tcall})


@typecheck(t=HailType)
def is_numeric(t) -> bool:
    return t.__class__ in _numeric_types


@typecheck(t=HailType)
def is_primitive(t) -> bool:
    return t.__class__ in _primitive_types


@typecheck(t=HailType)
def is_container(t) -> bool:
    return isinstance(t, (tarray, tdict, tset))


@typecheck(t=HailType)
def is_compound(t) -> bool:
    return (
        isinstance(t, (tndarray, tstruct, ttuple, tunion))
    )


def types_match(left, right) -> bool:
    return len(left) == len(right) and all(map(lambda lr: lr[0].dtype == lr[1].dtype, zip(left, right)))


def is_int32(x):
    return isinstance(x, (builtins.int, np.int32))


def is_int64(x):
    return isinstance(x, (builtins.int, np.int64))


def is_float32(x):
    return isinstance(x, (builtins.float, builtins.int, np.float32))


def is_float64(x):
    return isinstance(x, (builtins.float, builtins.int, np.float64))


def from_numpy(np_dtype):
    if np_dtype == np.int32:
        return tint32
    elif np_dtype == np.int64:
        return tint64
    elif np_dtype == np.float32:
        return tfloat32
    elif np_dtype == np.float64:
        return tfloat64
    elif np_dtype == np.bool_:
        return tbool
    else:
        raise ValueError(f"numpy type {np_dtype} could not be converted to a hail type.")


def dtypes_from_pandas(pd_dtype):
    if type(pd_dtype) == pd.StringDtype:
        return hl.tstr
    elif pd_dtype == np.int64:
        return hl.tint64
    elif pd_dtype == np.uint64:
        # Hail does *not* support unsigned integers but the next condition,
        # pd.api.types.is_integer_dtype(pd_dtype) would return true on unsigned 64-bit ints
        return None
    # For some reason pandas doesn't have `is_int32_dtype`, so we use `is_integer_dtype` if first branch failed.
    elif pd.api.types.is_integer_dtype(pd_dtype):
        return hl.tint32
    elif pd_dtype == np.float32:
        return hl.tfloat32
    elif pd_dtype == np.float64:
        return hl.tfloat64
    elif pd_dtype == bool:
        return hl.tbool
    return None


class tvariable(HailType):
    _cond_map: ClassVar = {
        'numeric': is_numeric,
        'int32': lambda x: x == tint32,
        'int64': lambda x: x == tint64,
        'float32': lambda x: x == tfloat32,
        'float64': lambda x: x == tfloat64,
        'locus': lambda x: isinstance(x, tlocus),
        'struct': lambda x: isinstance(x, tstruct),
        'union': lambda x: isinstance(x, tunion),
        'tuple': lambda x: isinstance(x, ttuple),
    }

    def __init__(self, name, cond):
        self.name = name
        self.cond = cond
        self.condf = tvariable._cond_map[cond] if cond else None
        self.box = Box.from_name(name)

    def unify(self, t):
        if self.condf and not self.condf(t):
            return False
        return self.box.unify(t)

    def clear(self):
        self.box.clear()

    def subst(self):
        return self.box.get()

    def __str__(self):
        s = '?' + self.name
        if self.cond:
            s = s + ':' + self.cond
        return s


_old_printer = pprint.PrettyPrinter


class TypePrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, HailType):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = TypePrettyPrinter  # monkey-patch pprint
