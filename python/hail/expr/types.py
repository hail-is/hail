import abc

import hail as hl
from hail.typecheck import *
from hail.utils import Struct
from hail.utils.java import scala_object, jset, jindexed_seq, Env, jarray_to_list, escape_parsable
from hail.genetics.reference_genome import reference_genome_type
from hail import genetics
from hail.expr.type_parsing import type_grammar, type_node_visitor
import json
from collections import Mapping


def dtype(type_str):
    r"""Parse a type from its string representation.

    Examples
    --------
    .. doctest::

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
    This function is able to reverse ``str(t)`` on a :class:`.Type`.

    The grammar is defined as follows:

    .. code-block:: text

        type = _ (array / set / dict / struct / tuple / interval / int64 / int32 / float32 / float64 / bool / str / call / str / locus) _
        int64 = "int64" / "tint64"
        int32 = "int32" / "tint32" / "int" / "tint"
        float32 = "float32" / "tfloat32"
        float64 = "float64" / "tfloat64" / "tfloat" / "float"
        bool = "tbool" / "bool"
        call = "tcall" / "call"
        str = "tstr" / "str"
        locus = ("tlocus" / "locus") _ "[" identifier "]"
        array = ("tarray" / "array") _ "<" type ">"
        set = ("tset" / "set") _ "<" type ">"
        dict = ("tdict" / "dict") _ "<" type "," type ">"
        struct = ("tstruct" / "struct") _ "{" (fields / _) "}"
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
    type_str : :obj:`str`
        String representation of type.

    Returns
    -------
    :class:`.Type`
    """
    tree = type_grammar.parse(type_str)
    return type_node_visitor.visit(tree)


class Type(object):
    """
    Hail type superclass.
    """

    def __init__(self):
        self._cached_jtype = None
        super(Type, self).__init__()

    def __repr__(self):
        s = str(self).replace("'", "\\'")
        return "dtype('{}')".format(s)

    @property
    def _jtype(self):
        if self._cached_jtype is None:
            self._cached_jtype = self._get_jtype()
        return self._cached_jtype

    @abc.abstractmethod
    def _eq(self, other):
        return

    def __eq__(self, other):
        return isinstance(other, Type) and self._eq(other)

    @abc.abstractmethod
    def __str__(self):
        return

    def __ne__(self, other):
        return not self.__eq__(other)

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
        :obj:`str`
        """
        l = []
        self._pretty(l, indent, increment)
        return ''.join(l)

    def _pretty(self, l, indent, increment):
        l.append(str(self))

    @classmethod
    def _from_java(cls, jtype):
        return hl.dtype(jtype.toString())

    @abc.abstractmethod
    def _typecheck(self, annotation):
        """
        Raise an exception if the given annotation is not the appropriate type.

        :param annotation: value to check
        """
        return

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

    def _convert_from_json_na(self, x):
        if x is None:
            return x
        else:
            return self._convert_from_json(x)

    def _convert_from_json(self, x):
        return x


class _tint32(Type):
    """Hail type for signed 32-bit integers.

    Their values can range from :math:`-2^{31}` to :math:`2^{31} - 1`
    (approximately 2.15 billion).

    In Python, these are represented as :obj:`int`.
    """

    def __init__(self):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TInt32Optional')
        super(_tint32, self).__init__()

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().makeInt(annotation)
        else:
            return None

    def _typecheck(self, annotation):
        if annotation is not None and not isinstance(annotation, int):
            raise TypeError("type 'tint32' expected Python 'int', but found type '%s'" % type(annotation))

    def __str__(self):
        return "int32"

    def _eq(self, other):
        return isinstance(other, _tint32)

    @property
    def min_value(self):
        return -(1 << 31)

    @property
    def max_value(self):
        return (1 << 31) - 1


class _tint64(Type):
    """Hail type for signed 64-bit integers.

    Their values can range from :math:`-2^{63}` to :math:`2^{63} - 1`.

    In Python, these are represented as :obj:`int`.
    """

    def __init__(self):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TInt64Optional')
        super(_tint64, self).__init__()

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        raise NotImplementedError('int64 conversion from Python to JVM')

    def _typecheck(self, annotation):
        if annotation and not isinstance(annotation, int):
            raise TypeError("type 'int64' expected Python 'int', but found type '%s'" % type(annotation))

    def __str__(self):
        return "int64"

    def _eq(self, other):
        return isinstance(other, _tint64)

    @property
    def min_value(self):
        return -(1 << 63)

    @property
    def max_value(self):
        return (1 << 63) - 1


class _tfloat32(Type):
    """Hail type for 32-bit floating point numbers.

    In Python, these are represented as :obj:`float`.
    """

    def __init__(self):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TFloat32Optional')
        super(_tfloat32, self).__init__()

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        # if annotation:
        #     return Env.jutils().makeFloat(annotation)
        # else:
        #     return annotation

        # FIXME: This function is unsupported until py4j-0.10.4: https://github.com/bartdag/py4j/issues/255
        raise NotImplementedError('float32 is currently unsupported in certain operations, use float64 instead')

    def _typecheck(self, annotation):
        if annotation is not None and not isinstance(annotation, (float, int)):
            raise TypeError("type 'float32' expected Python 'float', but found type '%s'" % type(annotation))

    def __str__(self):
        return "float32"

    def _eq(self, other):
        return isinstance(other, _tfloat32)

    def _convert_from_json(self, x):
        return float(x)


class _tfloat64(Type):
    """Hail type for 64-bit floating point numbers.

    In Python, these are represented as :obj:`float`.
    """

    def __init__(self):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TFloat64Optional')
        super(_tfloat64, self).__init__()

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().makeDouble(annotation)
        else:
            return None

    def _typecheck(self, annotation):
        if annotation is not None and not isinstance(annotation, (float, int)):
            raise TypeError("type 'float64' expected Python 'float', but found type '%s'" % type(annotation))
    def __str__(self):
        return "float64"

    def _eq(self, other):
        return isinstance(other, _tfloat64)

    def _convert_from_json(self, x):
        return float(x)

class _tstr(Type):
    """Hail type for text strings.

    In Python, these are represented as strings.
    """

    def __init__(self):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TStringOptional')
        super(_tstr, self).__init__()

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        return annotation

    def _typecheck(self, annotation):
        if annotation and not isinstance(annotation, str):
            raise TypeError("type 'str' expected Python 'str', but found type '%s'" % type(annotation))

    def __str__(self):
        return "str"

    def _eq(self, other):
        return isinstance(other, _tstr)


class _tbool(Type):
    """Hail type for Boolean (``True`` or ``False``) values.

    In Python, these are represented as :obj:`bool`.
    """

    def __init__(self):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TBooleanOptional')
        super(_tbool, self).__init__()

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        return annotation

    def _typecheck(self, annotation):
        if annotation is not None and not isinstance(annotation, bool):
            raise TypeError("type 'bool' expected Python 'bool', but found type '%s'" % type(annotation))

    def __str__(self):
        return "bool"

    def _eq(self, other):
        return isinstance(other, _tbool)


class tarray(Type):
    """Hail type for variable-length arrays of elements.

    In Python, these are represented as :obj:`list`.

    Notes
    -----
    Arrays contain elements of only one type, which is parameterized by
    `element_type`.

    Parameters
    ----------
    element_type : :class:`.Type`
        Element type of array.
    """

    @typecheck_method(element_type=Type)
    def __init__(self, element_type):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TArray').apply(element_type._jtype, False)
        self._element_type = element_type
        super(tarray, self).__init__()

    @property
    def element_type(self):
        """Array element type.

        Returns
        -------
        :class:`.Type`
            Element type.
        """
        return self._element_type

    def _convert_to_py(self, annotation):
        if annotation is not None:
            lst = Env.jutils().iterableToArrayList(annotation)
            return [self.element_type._convert_to_py(x) for x in lst]
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().arrayListToISeq(
                [self.element_type._convert_to_j(elt) for elt in annotation]
            )
        else:
            return None

    def _typecheck(self, annotation):
        if annotation is not None:
            if not isinstance(annotation, list):
                raise TypeError("type 'array' expected Python 'list', but found type '%s'" % type(annotation))
            for elt in annotation:
                self.element_type._typecheck(elt)

    def __str__(self):
        return "array<{}>".format(self.element_type)

    def _eq(self, other):
        return isinstance(other, tarray) and self.element_type == other.element_type

    def _pretty(self, l, indent, increment):
        l.append('array<')
        self.element_type._pretty(l, indent, increment)
        l.append('>')

    def _convert_from_json(self, x):
        return [self.element_type._convert_from_json_na(elt) for elt in x]

    def _convert_to_json(self, x):
        return [self.element_type._convert_to_json_na(elt) for elt in x]


class tset(Type):
    """Hail type for collections of distinct elements.

    In Python, these are represented as :obj:`set`.

    Notes
    -----
    Sets contain elements of only one type, which is parameterized by
    `element_type`.

    Parameters
    ----------
    element_type : :class:`.Type`
        Element type of set.
    """

    @typecheck_method(element_type=Type)
    def __init__(self, element_type):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TSet').apply(element_type._jtype, False)
        self._element_type = element_type
        super(tset, self).__init__()

    @property
    def element_type(self):
        """Set element type.

        Returns
        -------
        :class:`.Type`
            Element type.
        """
        return self._element_type

    def _convert_to_py(self, annotation):
        if annotation is not None:
            lst = Env.jutils().iterableToArrayList(annotation)
            return set([self.element_type._convert_to_py(x) for x in lst])
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return jset(
                [self.element_type._convert_to_j(elt) for elt in annotation]
            )
        else:
            return None

    def _typecheck(self, annotation):
        if annotation is not None:
            if not isinstance(annotation, set):
                raise TypeError("type 'set' expected Python 'set', but found type '%s'" % type(annotation))
            for elt in annotation:
                self.element_type._typecheck(elt)

    def __str__(self):
        return "set<{}>".format(self.element_type)

    def _eq(self, other):
        return isinstance(other, tset) and self.element_type == other.element_type

    def _pretty(self, l, indent, increment):
        l.append('set<')
        self.element_type._pretty(l, indent, increment)
        l.append('>')

    def _convert_from_json(self, x):
        return {self.element_type._convert_from_json_na(elt) for elt in x}

    def _convert_to_json(self, x):
        return [self.element_type._convert_to_json_na(elt) for elt in x]

class tdict(Type):
    """Hail type for key-value maps.

    In Python, these are represented as :obj:`dict`.

    Notes
    -----
    Dicts parameterize the type of both their keys and values with
    `key_type` and `value_type`.

    Parameters
    ----------
    key_type: :class:`.Type`
        Key type.
    value_type: :class:`.Type`
        Value type.
    """

    @typecheck_method(key_type=Type, value_type=Type)
    def __init__(self, key_type, value_type):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TDict').apply(
            key_type._jtype, value_type._jtype, False)
        self._key_type = key_type
        self._value_type = value_type
        super(tdict, self).__init__()

    @property
    def key_type(self):
        """Dict key type.

        Returns
        -------
        :class:`.Type`
            Key type.
        """
        return self._key_type

    @property
    def value_type(self):
        """Dict value type.

        Returns
        -------
        :class:`.Type`
            Value type.
        """
        return self._value_type

    def _convert_to_py(self, annotation):
        if annotation is not None:
            lst = Env.jutils().iterableToArrayList(annotation)
            d = dict()
            for x in lst:
                d[self.key_type._convert_to_py(x._1())] = self.value_type._convert_to_py(x._2())
            return d
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().javaMapToMap(
                {self.key_type._convert_to_j(k): self.value_type._convert_to_j(v) for k, v in annotation.items()}
            )
        else:
            return None

    def _typecheck(self, annotation):
        if annotation:
            if not isinstance(annotation, dict):
                raise TypeError("type 'dict' expected Python 'dict', but found type '%s'" % type(annotation))
            for k, v in annotation.items():
                self.key_type._typecheck(k)
                self.value_type._typecheck(v)

    def __str__(self):
        return "dict<{}, {}>".format(self.key_type, self.value_type)

    def _eq(self, other):
        return isinstance(other, tdict) and self.key_type == other.key_type and self.value_type == other.value_type

    def _pretty(self, l, indent, increment):
        l.append('dict<')
        self.key_type._pretty(l, indent, increment)
        l.append(', ')
        self.value_type._pretty(l, indent, increment)
        l.append('>')

    def _convert_from_json(self, x):
        return {self.key_type._convert_from_json_na(elt['key']): self.value_type._convert_from_json_na(elt['value']) for
                elt in x}

    def _convert_to_json(self, x):
        return [{'key': self.key_type._convert_to_json(k),
                 'value':self.value_type._convert_to_json(v)} for k, v in x.items()]


class tstruct(Type, Mapping):
    """Hail type for structured groups of heterogeneous fields.

    In Python, these are represented as :class:`.Struct`.

    Parameters
    ----------
    field_types : keyword args of :class:`.Type`
        Fields.
    """

    @typecheck_method(field_types=Type)
    def __init__(self, **field_types):
        self._field_types = field_types
        self._fields = tuple(field_types)

        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TStruct').apply(
            list(self._fields),
            [t._jtype for f, t in self._field_types.items()],
            False)

        super(tstruct, self).__init__()

    @property
    def fields(self):
        """Struct fields.

        Returns
        -------
        :obj:`tuple` of :class:`.Field`
            Struct fields.
        """
        return self._fields

    def _convert_to_py(self, annotation):
        if annotation is not None:
            d = dict()
            for i, (f, t) in enumerate(self.items()):
                d[f] = t._convert_to_py(annotation.get(i))
            return Struct(**d)
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return scala_object(Env.hail().annotations, 'Annotation').fromSeq(
                Env.jutils().arrayListToISeq(
                    [t._convert_to_j(annotation.get(f)) for f, t in self.items()]))
        else:
            return None

    def _typecheck(self, annotation):
        if annotation:
            if isinstance(annotation, Mapping):
                s = set(self)
                for f in annotation:
                    if f not in s:
                        raise TypeError("type '%s' expected fields '%s', but found fields '%s'" %
                                        (self, list(self), list(annotation)))
                for f, t in self.items():
                    t._typecheck(annotation.get(f))
            else:
                raise TypeError("type 'struct' expected type Mapping (e.g. hail.genetics.Struct or dict), but found '%s'" %
                                type(annotation))

    @typecheck_method(item=oneof(int, str))
    def __getitem__(self, item):
        if isinstance(item, str):
            return self._field_types[item]
        else:
            self._field_types[self._fields[item]]

    def __iter__(self):
        return iter(self._field_types)

    def __len__(self):
        return len(self._fields)

    def __str__(self):
        return "struct{{{}}}".format(
            ', '.join('{}: {}'.format(escape_parsable(f), str(t)) for f, t in self.items()))

    def _eq(self, other):
        return (isinstance(other, tstruct)
                and self._fields == other._fields
                and all(self[f] == other[f] for f in self._fields))

    def _pretty(self, l, indent, increment):
        pre_indent = indent
        indent += increment
        l.append('struct {')
        for i, (f, t) in enumerate(self.items()):
            if i > 0:
                l.append(', ')
            l.append('\n')
            l.append(' ' * indent)
            l.append('{}: '.format(escape_parsable(f)))
            t._pretty(l, indent, increment)
        l.append('\n')
        l.append(' ' * pre_indent)
        l.append('}')

    def _convert_from_json(self, x):
        return Struct(**{f: t._convert_from_json_na(x.get(f)) for f, t in self.items()})

    def _convert_to_json(self, x):
        return {f: t._convert_to_json_na(x[f]) for f, t in self.items()}

class ttuple(Type):
    """Hail type for tuples.

    In Python, these are represented as :obj:`tuple`.

    Parameters
    ----------
    types: varargs of :class:`.Type`
        Element types.
    """

    @typecheck_method(types=Type)
    def __init__(self, *types):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TTuple').apply(map(lambda t: t._jtype, types),
                                                                                      False)
        self._types = types
        super(ttuple, self).__init__()

    @property
    def types(self):
        """Tuple element types.

        Returns
        -------
        :obj:`tuple` of :class:`.Type`
        """
        return self._types

    def _convert_to_py(self, annotation):
        if annotation is not None:
            return tuple(*(t._convert_to_py(annotation.get(i)) for i, t in enumerate(self.types)))
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().arrayListToISeq(
                [self.types[i]._convert_to_j(elt) for i, elt in enumerate(annotation)]
            )
        else:
            return None

    def _typecheck(self, annotation):
        if annotation:
            if not isinstance(annotation, tuple):
                raise TypeError("type 'tuple' expected Python tuple, but found '%s'" %
                                type(annotation))
            if len(annotation) != len(self.types):
                raise TypeError("%s expected tuple of size '%i', but found '%s'" %
                                (self, len(self.types), annotation))
            for i, t in enumerate(self.types):
                t._typecheck((annotation[i]))

    def __str__(self):
        return "tuple({})".format(", ".join([str(t) for t in self.types]))

    def _eq(self, other):
        from operator import eq
        return isinstance(other, ttuple) and len(self.types) == len(other.types) and all(
            map(eq, self.types, other.types))

    def _pretty(self, l, indent, increment):
        pre_indent = indent
        indent += increment
        l.append('tuple (')
        for i, t in enumerate(self.types):
            if i > 0:
                l.append(', ')
            l.append('\n')
            l.append(' ' * indent)
            t._pretty(l, indent, increment)
        l.append('\n')
        l.append(' ' * pre_indent)
        l.append(')')

    def _convert_from_json(self, x):
        return tuple(self.types[i]._convert_from_json_na(x[i]) for i in range(len(self.types)))

    def _convert_to_json(self, x):
        return [self.types[i]._convert_to_json_na(x[i]) for i in range(len(self.types))]

class _tcall(Type):
    """Hail type for a diploid genotype.

    In Python, these are represented by :class:`.Call`.
    """

    def __init__(self):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TCallOptional')
        super(_tcall, self).__init__()

    @typecheck_method(annotation=nullable(int))
    def _convert_to_py(self, annotation):
        if annotation is not None:
            return genetics.Call._from_java(annotation)
        else:
            return None

    @typecheck_method(annotation=nullable(genetics.Call))
    def _convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._call
        else:
            return None

    def _typecheck(self, annotation):
        if annotation is not None and not isinstance(annotation, genetics.Call):
            raise TypeError("type 'call' expected Python hail.genetics.Call, but found %s'" %
                            type(annotation))

    def __str__(self):
        return "call"

    def _eq(self, other):
        return isinstance(other, _tcall)

    def _convert_from_json(self, x):
        return hl.Call._from_java(hl.Call._call_jobject().parse(x))

    def _convert_to_json(self, x):
        return str(x)


class tlocus(Type):
    """Hail type for a genomic coordinate with a contig and a position.

    In Python, these are represented by :class:`.Locus`.

    Parameters
    ----------
    reference_genome: :class:`.ReferenceGenome` or :obj:`str`
        Reference genome to use.
    """

    @typecheck_method(reference_genome=reference_genome_type)
    def __init__(self, reference_genome='default'):
        self._rg = reference_genome
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TLocus').apply(self._rg._jrep,
                                                                                      False)
        super(tlocus, self).__init__()

    def _convert_to_py(self, annotation):
        if annotation is not None:
            return genetics.Locus._from_java(annotation, self._rg)
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return None

    def _typecheck(self, annotation):
        if annotation is not None:
            if not isinstance(annotation, genetics.Locus):
                raise TypeError("type '{}' expected Python hail.genetics.Locus, but found '{}'"
                                .format(self, type(annotation)))
            if not self.reference_genome == annotation.reference_genome:
                raise TypeError("type '{}' encountered Locus with reference genome {}"
                                .format(self, repr(annotation.reference_genome)))

    def __str__(self):
        return "locus<{}>".format(escape_parsable(str(self.reference_genome)))

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

    def _pretty(self, l, indent, increment):
        l.append('locus<{}>'.format(escape_parsable(self.reference_genome.name)))

    def _convert_from_json(self, x):
        return genetics.Locus(x['contig'], x['position'], reference_genome=self.reference_genome)

    def _convert_to_json(self, x):
        return {'contig': x.contig, 'position': x.position}


class tinterval(Type):
    """Hail type for intervals of ordered values.

    In Python, these are represented by :class:`.Interval`.

    Notes
    -----
    Intervals are inclusive of the start point, but exclusive of the end point.

    Parameters
    ----------
    point_type: :class:`.Type`
        Interval point type.
    """

    @typecheck_method(point_type=Type)
    def __init__(self, point_type):
        self._get_jtype = lambda: scala_object(Env.hail().expr.types, 'TInterval').apply(self.point_type._jtype, False)
        self._point_type = point_type
        super(tinterval, self).__init__()

    @property
    def point_type(self):
        """Interval point type.

        Returns
        -------
        :class:`.Type`
            Interval point type.
        """
        return self._point_type

    def _convert_to_py(self, annotation):
        assert (isinstance(self._point_type, tlocus))
        if annotation is not None:
            return genetics.Interval._from_java(annotation, self._point_type.reference_genome)
        else:
            return None

    @typecheck_method(annotation=nullable(genetics.Interval))
    def _convert_to_j(self, annotation):
        assert (isinstance(self._point_type, tlocus))
        if annotation is not None:
            return annotation._jrep
        else:
            return None

    def _typecheck(self, annotation):
        assert (isinstance(self._point_type, tlocus))
        if annotation is not None and not isinstance(annotation, genetics.Interval):
            raise TypeError("type 'interval' expected Python hail.genetics.Interval, but found %s'" %
                            type(annotation))

    def __str__(self):
        return "interval<{}>".format(str(self.point_type))

    def _eq(self, other):
        return isinstance(other, tinterval) and self.point_type == other.point_type

    def _pretty(self, l, indent, increment):
        l.append('interval<')
        self.point_type._pretty(l, indent, increment)
        l.append('>')

    def _convert_from_json(self, x):
        if not isinstance(self.point_type, tlocus):
            raise NotImplementedError(self.point_type)
        return genetics.Interval(self.point_type._convert_from_json_na(x['start']),
                                 self.point_type._convert_from_json_na(x['end']))

    def _convert_to_json(self, x):
        if not isinstance(self.point_type, tlocus):
            raise NotImplementedError(self.point_type)
        return {'start': self.point_type._convert_to_json_na(x.start),
                'end': self.point_type._convert_to_json_na(x.end)}


tint32 = _tint32()
"""Hail type for signed 32-bit integers.

Their values can range from :math:`-2^{31}` to :math:`2^{31} - 1`
(approximately 2.15 billion).

In Python, these are represented as :obj:`int`.
"""


tint64 = _tint64()
"""Hail type for signed 64-bit integers.

Their values can range from :math:`-2^{63}` to :math:`2^{63} - 1`.

In Python, these are represented as :obj:`int`.
"""

tint = tint32
"""Alias for :py:data:`.tint32`."""

tfloat32 = _tfloat32()
"""Hail type for 32-bit floating point numbers.

In Python, these are represented as :obj:`float`.
"""

tfloat64 = _tfloat64()
"""Hail type for 64-bit floating point numbers.

In Python, these are represented as :obj:`float`.
"""

tfloat = tfloat64
"""Alias for :py:data:`.tfloat64`."""

tstr = _tstr()
"""Hail type for text strings.

In Python, these are represented as strings.
"""

tbool = _tbool()
"""Hail type for Boolean (``True`` or ``False``) values.

In Python, these are represented as :obj:`bool`.
"""

tcall = _tcall()
"""Hail type for a diploid genotype.

In Python, these are represented by :class:`.Call`.
"""

hts_entry_schema = tstruct(GT=tcall, AD=tarray(tint32), DP=tint32, GQ=tint32, PL=tarray(tint32))

_numeric_types = {tint32, tint64, tfloat32, tfloat64}
_primitive_types = _numeric_types.union({tbool, tstr})


@typecheck(t=Type)
def is_numeric(t):
    return t in _numeric_types


@typecheck(t=Type)
def is_primitive(t):
    return t in _primitive_types

@typecheck(t=Type)
def is_container(t):
    return (isinstance(t, tarray)
            or isinstance(t, tset)
            or isinstance(t, tdict)
            or isinstance(t, ttuple)
            or isinstance(t, tstruct))

import pprint

_old_printer = pprint.PrettyPrinter


class TypePrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, Type):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = TypePrettyPrinter  # monkey-patch pprint
