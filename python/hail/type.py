from hail.java import scala_object, scala_package_object, Env
from hail.representation import Variant, AltAllele, Genotype, Locus, Interval, Struct

class Type(object):
    """Type of values."""

    def __init__(self, jtype):
        self._jtype = jtype

    def __repr__(self):
        return self._jtype.toString()

    def __str__(self):
        return self._jtype.toPrettyString(False, False)

    def __eq__(self, other):
        return self._jtype.equals(other._jtype)

    @classmethod
    def _from_java(cls, jtype):
        class_name = jtype.getClass().getCanonicalName()
        if class_name == 'is.hail.expr.TInt$':
            return TInt()
        elif class_name == 'is.hail.expr.TLong$':
            return TLong()
        elif class_name == 'is.hail.expr.TFloat$':
            return TFloat()
        elif class_name == 'is.hail.expr.TDouble$':
            return TDouble()
        elif class_name == 'is.hail.expr.TString$':
            return TString()
        elif class_name == 'is.hail.expr.TBoolean$':
            return TBoolean()
        elif class_name == 'is.hail.expr.TSample$':
            return TString()
        elif class_name == 'is.hail.expr.TArray':
            return TArray._from_java(jtype)
        elif class_name == 'is.hail.expr.TSet':
            return TSet._from_java(jtype)
        elif class_name == 'is.hail.expr.TDict':
            return TDict._from_java(jtype)
        elif class_name == 'is.hail.expr.TStruct':
            return TStruct._from_java(jtype)
        elif class_name == 'is.hail.expr.TVariant$':
            return TVariant()
        elif class_name == 'is.hail.expr.TAltAllele$':
            return TAltAllele()
        elif class_name == 'is.hail.expr.TLocus$':
            return TLocus()
        elif class_name == 'is.hail.expr.TInterval$':
            return TInterval()
        elif class_name == 'is.hail.expr.TGenotype$':
            return TGenotype()
        else:
            raise TypeError("unknown type class: '%s'" % class_name)


class TInt(Type):
    """
    python type: int
    """

    def __init__(self):
        super(TInt, self).__init__(scala_object(Env.hail_package().expr, 'TInt'))

    def convert(self, annotation):
        return annotation


class TLong(Type):
    """
    python type: long
    """

    def __init__(self):
        super(TLong, self).__init__(scala_object(Env.hail_package().expr, 'TLong'))

    def convert(self, annotation):
        return annotation


class TFloat(Type):
    """
    python type: float
    """

    def __init__(self):
        super(TFloat, self).__init__(scala_object(Env.hail_package().expr, 'TFloat'))

    def convert(self, annotation):
        return annotation


class TDouble(Type):
    """
    python type: float
    """

    def __init__(self):
        super(TDouble, self).__init__(scala_object(Env.hail_package().expr, 'TDouble'))

    def convert(self, annotation):
        return annotation


class TString(Type):
    """
    python type: str
    """

    def __init__(self):
        super(TString, self).__init__(scala_object(Env.hail_package().expr, 'TString'))

    def convert(self, annotation):
        return annotation

class TBoolean(Type):
    """
    python type: bool
    """

    def __init__(self):
        super(TBoolean, self).__init__(scala_object(Env.hail_package().expr, 'TBoolean'))

    def convert(self, annotation):
        return annotation


class TArray(Type):
    """
    python type: list
    """

    def __init__(self, element_type):
        """
        :param :class:`.Type` element_type: Hail type of array element
        """
        jtype = scala_object(Env.hail_package().expr, 'TArray').apply(element_type._jtype)
        self.element_type = element_type
        super(TArray, self).__init__(jtype)

    @classmethod
    def _from_java(cls, jtype):
        t = TArray.__new__(cls)
        t.element_type = Type._from_java(jtype.elementType())
        t._jtype = jtype
        return t

    def convert(self, annotation):
        if annotation:
            lst = scala_package_object(Env.hail_package().utils).iterableToArrayList(annotation)
            return [self.element_type.convert(x) for x in lst]
        else:
            return annotation


class TSet(Type):
    """
    python type: set
    """

    def __init__(self, element_type):
        """
        :param :class:`.Type` element_type: Hail type of set element
        """
        jtype = scala_object(Env.hail_package().expr, 'TSet').apply(element_type._jtype)
        self.element_type = element_type
        super(TSet, self).__init__(jtype)

    @classmethod
    def _from_java(cls, jtype):
        t = TSet.__new__(cls)
        t.element_type = Type._from_java(jtype.elementType())
        t._jtype = jtype
        return t

    def convert(self, annotation):
        if annotation:
            lst = scala_package_object(Env.hail_package().utils).iterableToArrayList(annotation)
            return set([self.element_type.convert(x) for x in lst])
        else:
            return annotation


class TDict(Type):
    """
    python type: dict
    """

    def __init__(self, element_type):
        """
        :param :class:`.Type` element_type: Hail type of dict element
        """
        jtype = scala_object(Env.hail_package().expr, 'TDict').apply(element_type._jtype)
        self.element_type = element_type
        super(TDict, self).__init__(jtype)

    @classmethod
    def _from_java(cls, jtype):
        t = TDict.__new__(cls)
        t.element_type = Type._from_java(jtype.elementType())
        t._jtype = jtype
        return t

    def convert(self, annotation):
        if annotation:
            lst = scala_package_object(Env.hail_package().utils).iterableToArrayList(annotation)
            d = dict()
            for x in lst:
                d[x._1()] = self.element_type.convert(x._2())
            return d
        else:
            return annotation


class Field(object):
    def __init__(self, name, htype):
        self.name = name
        self.type = htype


class TStruct(Type):
    """
    python type: :class:`hail.representation.Struct`
    """

    def __init__(self, names, types):
        """
        :param names: names of fields
        :type names: list of str
        :param types: types of fields
        :type types: list of :class:`.Type`
        """

        if len(names) != len(types):
            raise ValueError('length of names and types not equal: %d and %d' % (len(names), len(types)))
        jtype = scala_object(Env.hail_package().expr, 'TStruct').apply(names, map(lambda t: t._jtype, types))
        self.fields = [Field(names[i], types[i]) for i in xrange(len(names))]

        super(TStruct, self).__init__(jtype)

    @classmethod
    def _from_java(cls, jtype):
        struct = TStruct.__new__(cls)
        struct._init_from_java(jtype)
        struct._jtype = jtype
        return struct

    def _init_from_java(self, jtype):

        jfields = scala_package_object(Env.hail_package().utils).iterableToArrayList(jtype.fields())
        self.fields = [Field(f.name(), Type._from_java(f.typ())) for f in jfields]

    def convert(self, annotation):
        if annotation:
            d = dict()
            for i, f in enumerate(self.fields):
                d[f.name] = f.type.convert(annotation.get(i))
            return Struct(d)
        else:
            return annotation


class TVariant(Type):
    """
    python type: :class:`hail.representation.Variant`
    """

    def __init__(self):
        super(TVariant, self).__init__(scala_object(Env.hail_package().expr, 'TVariant'))

    def convert(self, annotation):
        if annotation:
            return Variant._from_java(annotation)
        else:
            return annotation


class TAltAllele(Type):
    """
    python type: :class:`hail.representation.AltAllele`
    """

    def __init__(self):
        super(TAltAllele, self).__init__(scala_object(Env.hail_package().expr, 'TAltAllele'))

    def convert(self, annotation):
        if annotation:
            return AltAllele._from_java(annotation)
        else:
            return annotation


class TGenotype(Type):
    """
    python type: :class:`hail.representation.Genotype`
    """

    def __init__(self):
        super(TGenotype, self).__init__(scala_object(Env.hail_package().expr, 'TGenotype'))

    def convert(self, annotation):
        if annotation:
            return Genotype._from_java(annotation)
        else:
            return annotation


class TLocus(Type):
    """
    python type: :class:`hail.representation.Locus`
    """

    def __init__(self):
        super(TLocus, self).__init__(scala_object(Env.hail_package().expr, 'TLocus'))

    def convert(self, annotation):
        if annotation:
            return Locus._from_java(annotation)
        else:
            return annotation


class TInterval(Type):
    """
    python type: :class:`hail.representation.Interval`
    """

    def __init__(self):
        super(TInterval, self).__init__(scala_object(Env.hail_package().expr, 'TInterval'))

    def convert(self, annotation):
        if annotation:
            return Interval._from_java(annotation)
        else:
            return annotation
