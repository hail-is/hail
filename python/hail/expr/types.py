import abc

from hail.history import *
from hail.typecheck import *
from hail.utils import Struct
from hail.utils.java import scala_object, jset, jindexed_seq, Env
import hail.genetics as genetics


class TypeCheckError(Exception):
    """
    Error thrown at mismatch between expected and supplied python types.

    :param str message: Error message
    """

    def __init__(self, message):
        self.msg = message
        super(TypeCheckError).__init__(TypeCheckError)

    def __str__(self):
        return self.msg


class Type(HistoryMixin):
    """
    Hail type superclass used for annotations and expression language.
    """
    __metaclass__ = abc.ABCMeta

    _hts_schema = None

    @classmethod
    def hts_schema(cls):
        """
        The high-through sequencing (HTS) genotype schema:

        .. code-block:: text

          Struct {
            GT: Call,
            AD: Array[!Int32],
            DP: Int32,
            GQ: Int32,
            PL: Array[!Int32].
          }
        """

        if not cls._hts_schema:
            cls._hts_schema = TStruct(
                ['GT', 'AD', 'DP', 'GQ', 'PL'],
                [TCall(), TArray(TInt32(True)), TInt32(), TInt32(), TArray(TInt32(True))])
        return cls._hts_schema

    def __init__(self, jtype):
        self._jtype = jtype
        self.required = jtype.required()
        super(Type, self).__init__()

    def __repr__(self):
        return ("!" if self.required else "") + self._repr()

    def _repr(self):
        return self._jtype.toString()

    def __str__(self):
        return self._jtype.toString()

    def __eq__(self, other):
        return isinstance(other, Type) and self._jtype.equals(other._jtype)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self._jtype.hashCode()

    def pretty(self):
        """Returns a prettily formatted string representation of the type.

        :param int indent: Number of spaces to indent.

        :rtype: str
        """

        return self._jtype.toPrettyString(False)

    @classmethod
    def _from_java(cls, jtype):
        # FIXME string matching is pretty hacky
        class_name = jtype.getClass().getCanonicalName()

        if class_name in _intern_classes:
            type, required = _intern_classes[class_name]
            return type(required)
        elif class_name == 'is.hail.expr.types.TArray':
            return TArray._from_java(jtype)
        elif class_name == 'is.hail.expr.types.TSet':
            return TSet._from_java(jtype)
        elif class_name == 'is.hail.expr.types.TDict':
            return TDict._from_java(jtype)
        elif class_name == 'is.hail.expr.types.TStruct':
            return TStruct._from_java(jtype)
        elif class_name == 'is.hail.expr.types.TVariant':
            return TVariant._from_java(jtype)
        elif class_name == 'is.hail.expr.types.TLocus':
            return TLocus._from_java(jtype)
        elif class_name == 'is.hail.expr.types.TInterval':
            return TInterval._from_java(jtype)
        else:
            raise TypeError("unknown type class: '%s'" % class_name)

    @abc.abstractmethod
    def _typecheck(self, annotation):
        """
        Raise an exception if the given annotation is not the appropriate type.

        :param annotation: value to check
        """
        return

    def _deep_optional(self):
        return Type._from_java(self._jtype.deepOptional())


class Intern(type):
    _instances = {}

    def __call__(cls, required=False):
        p = (cls, required)
        if p not in cls._instances:
            cls._instances[p] = super(Intern, cls).__call__(required=required)
        return cls._instances[p]


class InternType(Intern, abc.ABCMeta):
    pass


class TInt32(Type):
    """Hail type for signed 32-bit integers.

    Their values can range from :math:`-2^{31}` to :math:`2^{31} - 1`
    (approximately 2.15 billion).

    In Python, these are represented as :obj:`int`.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TInt32, self).__init__(scala_object(Env.hail().expr.types, 'TInt32').apply(required))

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().makeInt(annotation)
        else:
            return None

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError(self.__repr__+" can't have missing annotation")
        if annotation and not isinstance(annotation, int):
            raise TypeCheckError("TInt32 expected type 'int', but found type '%s'" % type(annotation))

    def _repr(self):
        return "TInt32()"

class TInt64(Type):
    """Hail type for signed 64-bit integers.

    Their values can range from :math:`-2^{63}` to :math:`2^{63} - 1`.

    In Python, these are represented as :obj:`int`.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TInt64, self).__init__(scala_object(Env.hail().expr.types, 'TInt64').apply(required))
        self.required = required

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().makeLong(annotation)
        else:
            return None

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError(self.__repr__+" can't have missing annotation")
        if annotation and not (isinstance(annotation, long) or isinstance(annotation, int)):
            raise TypeCheckError("TInt64 expected type 'int' or 'long', but found type '%s'" % type(annotation))

    def _repr(self):
        return "TInt64()"


class TFloat32(Type):
    """Hail type for 32-bit floating point numbers.

    In Python, these are represented as :obj:`float`.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TFloat32, self).__init__(scala_object(Env.hail().expr.types, 'TFloat32').apply(required))

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        # if annotation:
        #     return Env.jutils().makeFloat(annotation)
        # else:
        #     return annotation

        # FIXME: This function is unsupported until py4j-0.10.4: https://github.com/bartdag/py4j/issues/255
        raise NotImplementedError('TFloat32 is currently unsupported in certain operations, use TFloat64 instead')

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError(self.__repr__+" can't have missing annotation")
        if annotation and not isinstance(annotation, float):
            raise TypeCheckError("TFloat32 expected type 'float', but found type '%s'" % type(annotation))

    def _repr(self):
        return "TFloat32()"


class TFloat64(Type):
    """Hail type for 64-bit floating point numbers.

    In Python, these are represented as :obj:`float`.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TFloat64, self).__init__(scala_object(Env.hail().expr.types, 'TFloat64').apply(required))

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return Env.jutils().makeDouble(annotation)
        else:
            return None

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError(self.__repr__+" can't have missing annotation")
        if annotation and not isinstance(annotation, float):
            raise TypeCheckError("TFloat64 expected type 'float', but found type '%s'" % type(annotation))

    def _repr(self):
        return "TFloat64()"


class TString(Type):
    """Hail type for text strings.

    In Python, these are represented as :obj:`unicode` strings.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TString, self).__init__(scala_object(Env.hail().expr.types, 'TString').apply(required))

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        return annotation

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError(self.__repr__+" can't have missing annotation")
        if annotation and not (isinstance(annotation, str) or isinstance(annotation, unicode)):
            raise TypeCheckError("TString expected type 'str', but found type '%s'" % type(annotation))

    def _repr(self):
        return "TString()"


class TBoolean(Type):
    """Hail type for Boolean (``True`` or ``False``) values.

    In Python, these are represented as :obj:`bool`.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TBoolean, self).__init__(scala_object(Env.hail().expr.types, 'TBoolean').apply(required))

    def _convert_to_py(self, annotation):
        return annotation

    def _convert_to_j(self, annotation):
        return annotation

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError(self.__repr__+" can't have missing annotation")
        if annotation and not isinstance(annotation, bool):
            raise TypeCheckError("TBoolean expected type 'bool', but found type '%s'" % type(annotation))

    def _repr(self):
        return "TBoolean()"


class TArray(Type):
    """Hail type for variable-length arrays of elements.

    In Python, these are represented as :obj:`list`.

    Note
    ----
    Arrays contain elements of only one type, which is parameterized by
    `element_type`.

    Parameters
    ----------
    element_type : :class:`.Type`
        Element type of array.
    """

    @record_init
    def __init__(self, _element_type, required=False):
        jtype = scala_object(Env.hail().expr.types, 'TArray').apply(_element_type._jtype, required)
        self._element_type = _element_type
        super(TArray, self).__init__(jtype)

    @property
    def element_type(self):
        """Array element type.

        Returns
        -------
        :class:`.Type`
            Element type.
        """
        return self._element_type

    @classmethod
    def _from_java(cls, jtype):
        t = TArray.__new__(cls)
        t._element_type = Type._from_java(jtype.elementType())
        t._jtype = jtype
        t.required = jtype.required()
        super(Type, t).__init__()
        return t

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
        if annotation:
            if not isinstance(annotation, list):
                raise TypeCheckError("TArray expected type 'list', but found type '%s'" % type(annotation))
            for elt in annotation:
                self.element_type._typecheck(elt)
        elif self.required:
            raise TypeCheckError("!TArray() can't have missing annotation")

    def _repr(self):
        return "TArray({})".format(repr(self.element_type))


class TSet(Type):
    """Hail type for collections of distinct elements.

    In Python, these are represented as :obj:`set`.

    Note
    ----
    Sets contain elements of only one type, which is parameterized by
    `element_type`.

    Parameters
    ----------
    element_type : :class:`.Type`
        Element type of set.
    """

    @record_init
    def __init__(self, element_type, required=False):
        """
        :param :class:`.Type` element_type: Hail type of set element
        """
        jtype = scala_object(Env.hail().expr.types, 'TSet').apply(element_type._jtype, required)
        self._element_type = element_type
        super(TSet, self).__init__(jtype)

    @property
    def element_type(self):
        """Set element type.

        Returns
        -------
        :class:`.Type`
            Element type.
        """
        return self._element_type

    @classmethod
    def _from_java(cls, jtype):
        t = TSet.__new__(cls)
        t._element_type = Type._from_java(jtype.elementType())
        t._jtype = jtype
        t.required = jtype.required()
        super(Type, t).__init__()
        return t

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
        if annotation:
            if not isinstance(annotation, set):
                raise TypeCheckError("TSet expected type 'set', but found type '%s'" % type(annotation))
            for elt in annotation:
                self.element_type._typecheck(elt)

        elif self.required:
            raise TypeCheckError("!TSet() can't have missing annotation")

    def _repr(self):
        return "TSet({})".format(repr(self.element_type))


class TDict(Type):
    """Hail type for key-value maps.

    In Python, these are represented as :obj:`dict`.

    Note
    ----
    Dicts parameterize the type of both their keys and values with
    `key_type` and `value_type`.

    Parameters
    ----------
    key_type: :class:`.Type`
        Key type.
    value_type: :class:`.Type`
        Value type.
    """

    @record_init
    def __init__(self, key_type, value_type, required=False):
        jtype = scala_object(Env.hail().expr.types, 'TDict').apply(key_type._jtype, value_type._jtype, required)
        self._key_type = key_type
        self._value_type = value_type
        super(TDict, self).__init__(jtype)

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

    @classmethod
    def _from_java(cls, jtype):
        t = TDict.__new__(cls)
        t._key_type = Type._from_java(jtype.keyType())
        t._value_type = Type._from_java(jtype.valueType())
        t._jtype = jtype
        t.required = jtype.required()
        super(Type, t).__init__()
        return t

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
                {self.key_type._convert_to_j(k): self.value_type._convert_to_j(v) for k, v in annotation.iteritems()}
            )
        else:
            return None

    def _typecheck(self, annotation):
        if annotation:
            if not isinstance(annotation, dict):
                raise TypeCheckError("TDict expected type 'dict', but found type '%s'" % type(annotation))
            for k, v in annotation.iteritems():
                self.key_type._typecheck(k)
                self.value_type._typecheck(v)
        elif self.required:
            raise TypeCheckError("!TDict() can't have missing annotation")


    def _repr(self):
        return "TDict({}, {})".format(repr(self.key_type), repr(self.value_type))

class Field(object):
    """Class representing a field of a :class:`.TStruct`."""

    def __init__(self, name, typ):
        self._name = name
        self._typ = typ

    @property
    def name(self):
        """Field name.

        Returns
        -------
        :obj:`str`
            Field name.
        """
        return self._name

    @property
    def typ(self):
        """Field type.

        Returns
        -------
        :class:`.Type`
            Field type.
        """
        return self._typ

class TStruct(Type):
    """Hail type for structured groups of heterogeneous fields.

    In Python, these are represented as :class:`.Struct`.

    Parameters
    ----------
    names: :obj:`list` of :obj:`str`
        Field names.
    types: :obj:`list` of :class:`.Type`
        Field types.
    """

    @typecheck_method(names=listof(strlike), types=listof(Type), required=bool)
    def __init__(self, names, types, required=False):

        if len(names) != len(types):
            raise ValueError('length of names and types not equal: %d and %d' % (len(names), len(types)))
        jtype = scala_object(Env.hail().expr.types, 'TStruct').apply(names, map(lambda t: t._jtype, types), required)
        self._fields = [Field(names[i], types[i]) for i in range(len(names))]

        super(TStruct, self).__init__(jtype)

    @property
    def fields(self):
        """Struct fields.

        Returns
        -------
        :obj:`list` of :class:`.Field`
            Struct fields.
        """
        return self._fields

    @classmethod
    def from_fields(cls, fields, required=False):
        """Creates a new TStruct from field objects.

        :param fields: The TStruct fields.
        :type fields: list of :class:`.Field`

        :param bool required: Flag for whether the struct can be missing.

        :return: TStruct from input fields
        :rtype: :class:`.TStruct`
        """

        struct = TStruct.__new__(cls)
        struct._fields = fields
        jfields = [scala_object(Env.hail().expr.types, 'Field').apply(f.name, f.typ._jtype, i) for i, f in enumerate(fields)]
        jtype = scala_object(Env.hail().expr.types, 'TStruct').apply(jindexed_seq(jfields), required)
        return TStruct._from_java(jtype)

    @classmethod
    def _from_java(cls, jtype):
        struct = TStruct.__new__(cls)
        struct._init_from_java(jtype)
        struct._jtype = jtype
        struct.required = jtype.required()
        super(Type, struct).__init__()
        return struct

    def _init_from_java(self, jtype):

        jfields = Env.jutils().iterableToArrayList(jtype.fields())
        self._fields = [Field(f.name(), Type._from_java(f.typ())) for f in jfields]

    def _convert_to_py(self, annotation):
        if annotation is not None:
            d = dict()
            for i, f in enumerate(self.fields):
                d[f.name] = f.typ._convert_to_py(annotation.get(i))
            return Struct(**d)
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return scala_object(Env.hail().annotations, 'Annotation').fromSeq(
                Env.jutils().arrayListToISeq(
                    [f.typ._convert_to_j(annotation.get(f.name)) for f in self.fields]
                )
            )
        else:
            return None

    def _typecheck(self, annotation):
        if annotation:
            if not isinstance(annotation, Struct):
                raise TypeCheckError("TStruct expected type hail.genetics.Struct, but found '%s'" %
                                     type(annotation))
            for f in self.fields:
                if not (f.name in annotation):
                    raise TypeCheckError("TStruct expected fields '%s', but found fields '%s'" %
                                         ([f.name for f in self.fields], annotation._fields))
                f.typ._typecheck((annotation[f.name]))
        elif self.required:
            raise TypeCheckError("!TStruct cannot be missing")

    def __repr__(self):
        names = [fd.name for fd in self.fields]
        types = [fd.typ for fd in self.fields]
        return "TStruct({}, {})".format(repr(list(names)), repr(list(types)))

    def _merge(self, other):
        return TStruct._from_java(self._jtype.merge(other._jtype)._1())

    def _drop(self, *identifiers):
        return TStruct._from_java(self._jtype.filter(jset(list(identifiers)), False)._1())

    def _select(self, *identifiers):
        return TStruct._from_java(self._jtype.filter(jset(list(identifiers)), True)._1())


class TVariant(Type):
    """Hail type for a genomic variant with a coordinate and list of alleles.

    In Python, these are represented by :class:`.Variant`.

    Parameters
    ----------
    reference_genome: :class:`.GenomeReference`
        Reference genome to use. Default is
        :meth:`hail.api1.HailContext.default_reference`.
    """

    @record_init
    @typecheck_method(reference_genome=nullable(genetics.GenomeReference),
                      required=bool)
    def __init__(self, reference_genome=None, required=False):
        self._rg = reference_genome if reference_genome else Env.hc().default_reference
        jtype = scala_object(Env.hail().expr.types, 'TVariant').apply(self._rg._jrep, required)
        super(TVariant, self).__init__(jtype)

    @classmethod
    def _from_java(cls, jtype):
        v = TVariant.__new__(cls)
        v._jtype = jtype
        v._rg = genetics.GenomeReference._from_java(jtype.gr())
        v.required = jtype.required()
        super(Type, v).__init__()
        return v

    def _convert_to_py(self, annotation):
        if annotation is not None:
            return genetics.Variant._from_java(annotation, self._rg)
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return None

    def _typecheck(self, annotation):
        if annotation and not isinstance(annotation, genetics.Variant):
            raise TypeCheckError('TVariant expected type hail.genetics.Variant, but found %s' %
                                 type(annotation))

    def _repr(self):
        return "TVariant()"

    @property
    @record_property
    def reference_genome(self):
        """Reference genome.

        Returns
        -------
        :class:`.GenomeReference`
            Reference genome.
        """
        return self._rg


class TAltAllele(Type):
    """Hail type for a reference/alternate polymorphism.

    In Python, these are represented by :class:`.AltAllele`.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TAltAllele, self).__init__(scala_object(Env.hail().expr.types, 'TAltAllele').apply(required))

    def _convert_to_py(self, annotation):
        if annotation is not None:
            return genetics.AltAllele._from_java(annotation)
        else:
            return None

    def _convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return None

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError('!TAltAllele cannot be missing')
        if annotation and not isinstance(annotation, genetics.AltAllele):
            raise TypeCheckError('TAltAllele expected type hail.genetics.AltAllele, but found %s' %
                                 type(annotation))

    def _repr(self):
        return "TAltAllele()"


class TCall(Type):
    """Hail type for a diploid genotype.

    In Python, these are represented by :class:`.Call`.
    """
    __metaclass__ = InternType

    @record_init
    def __init__(self, required=False):
        super(TCall, self).__init__(scala_object(Env.hail().expr.types, 'TCall').apply(required))

    @typecheck_method(annotation=nullable(integral))
    def _convert_to_py(self, annotation):
        if annotation is not None:
            return genetics.Call(annotation)
        else:
            return None

    @typecheck_method(annotation=nullable(genetics.Call))
    def _convert_to_j(self, annotation):
        if annotation is not None:
            return annotation.gt
        else:
            return None

    def _typecheck(self, annotation):
        if not annotation and self.required:
            raise TypeCheckError('!TCall cannot be missing')
        if annotation and not isinstance(annotation, genetics.Call):
            raise TypeCheckError('TCall expected type hail.genetics.Call, but found %s' %
                                 type(annotation))

    def _repr(self):
        return "TCall()"


class TLocus(Type):
    """Hail type for a genomic coordinate with a contig and a position.

    In Python, these are represented by :class:`.Locus`.

    Parameters
    ----------
    reference_genome: :class:`.GenomeReference`
        Reference genome to use. Default is
        :meth:`hail.api1.HailContext.default_reference`.
    """

    @record_init
    @typecheck_method(reference_genome=nullable(genetics.GenomeReference),
                      required=bool)
    def __init__(self, reference_genome=None, required=False):
        self._rg = reference_genome if reference_genome else Env.hc().default_reference
        jtype = scala_object(Env.hail().expr.types, 'TLocus').apply(self._rg._jrep, required)
        super(TLocus, self).__init__(jtype)

    @classmethod
    def _from_java(cls, jtype):
        l = TLocus.__new__(cls)
        l._jtype = jtype
        l._rg = genetics.GenomeReference._from_java(jtype.gr())
        l.required = jtype.required()
        super(Type, l).__init__()
        return l

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
        if not annotation and self.required:
            raise TypeCheckError('!TLocus cannot be missing')
        if annotation and not isinstance(annotation, genetics.Locus):
            raise TypeCheckError('TLocus expected type hail.genetics.Locus, but found %s' %
                                 type(annotation))

    def _repr(self):
        return "TLocus()"

    @property
    @record_property
    def reference_genome(self):
        """Reference genome.

        Returns
        -------
        :class:`.GenomeReference`
            Reference genome.
        """
        return self._rg


class TInterval(Type):
    """Hail type for intervals of ordered values.

    In Python, these are represented by :class:`.Interval`.

    Note
    ----
    Intervals are inclusive of the start point, but exclusive of the end point.

    Parameters
    ----------
    point_type: :class:`.Type`
        Interval point type.
    """

    @record_init
    @typecheck_method(point_type=Type,
                      required=bool)
    def __init__(self, point_type, required=False):
        jtype = scala_object(Env.hail().expr.types, 'TInterval').apply(point_type._jtype, required)
        self._point_type = point_type
        super(TInterval, self).__init__(jtype)

    @property
    def point_type(self):
        """Interval point type.

        Returns
        -------
        :class:`.Type`
            Interval point type.
        """
        return self._point_type

    @classmethod
    def _from_java(cls, jtype):
        i = TInterval.__new__(cls)
        i._point_type = Type._from_java(jtype.pointType())
        i._jtype = jtype
        i.required = jtype.required()
        super(Type, i).__init__()
        return i

    def _convert_to_py(self, annotation):
        assert(isinstance(self._point_type, TLocus))
        if annotation is not None:
            return genetics.Interval._from_java(annotation, self._point_type.reference_genome)
        else:
            return None

    @typecheck_method(annotation=nullable(genetics.Interval))
    def _convert_to_j(self, annotation):
        assert(isinstance(self._point_type, TLocus))
        if annotation is not None:
            return annotation._jrep
        else:
            return None

    def _typecheck(self, annotation):
        assert(isinstance(self._point_type, TLocus))
        if not annotation and self.required:
            raise TypeCheckError('!TInterval is a required field')
        if annotation and not isinstance(annotation, genetics.Interval):
            raise TypeCheckError('TInterval expected type hail.genetics.Interval, but found %s' %
                                 type(annotation))

    def _repr(self):
        return "TInterval({})".format(repr(self._point_type))

_intern_classes = {'is.hail.expr.types.TInt32Optional$': (TInt32, False),
                   'is.hail.expr.types.TInt32Required$': (TInt32, True),
                   'is.hail.expr.types.TInt64Optional$': (TInt64, False),
                   'is.hail.expr.types.TInt64Required$': (TInt64, True),
                   'is.hail.expr.types.TFloat32Optional$': (TFloat32, False),
                   'is.hail.expr.types.TFloat32Required$': (TFloat32, True),
                   'is.hail.expr.types.TFloat64Optional$': (TFloat64, False),
                   'is.hail.expr.types.TFloat64Required$': (TFloat64, True),
                   'is.hail.expr.types.TBooleanOptional$': (TBoolean, False),
                   'is.hail.expr.types.TBooleanRequired$': (TBoolean, True),
                   'is.hail.expr.types.TStringOptional$': (TString, False),
                   'is.hail.expr.types.TStringRequired$': (TString, True),
                   'is.hail.expr.types.TAltAlleleOptional$': (TAltAllele, False),
                   'is.hail.expr.types.TAltAlleleRequired$': (TAltAllele, True),
                   'is.hail.expr.types.TCallOptional$': (TCall, False),
                   'is.hail.expr.types.TCallRequired$': (TCall, True)}


@typecheck(t=Type)
def is_numeric(t):
    return isinstance(t, TInt32) or isinstance(t, TInt64) or isinstance(t, TFloat32) or isinstance(t, TFloat64)


import pprint

_old_printer = pprint.PrettyPrinter


class TypePrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, Type):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = TypePrettyPrinter  # monkey-patch pprint
