# from hail.context import HailContext
from hail.java import scala_object
from hail.representation import Struct


class Type(object):
    """Type of values."""

    def __init__(self, jtype):
        self._jtype = jtype

    def __repr__(self):
        return self._jtype.toString()

    def __str__(self):
        return self._jtype.toPrettyString(False, False)

    @classmethod
    def _from_java(cls, jtype):
        # if switch on jtype here
        pass


class TInt(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TInt'))

    @classmethod
    def _from_java(cls, jtype):
        return TInt()

    def convert(self, annotation):
        return annotation


class TLong(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TLong'))

    @classmethod
    def _from_java(cls, jtype):
        return TLong()

    def convert(self, annotation):
        return annotation


class TFloat(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TFloat'))

    @classmethod
    def _from_java(cls, jtype):
        return TFloat()

    def convert(self, annotation):
        return annotation


class TDouble(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TDouble'))

    @classmethod
    def _from_java(cls, jtype):
        return TDouble()

    def convert(self, annotation):
        return annotation


class TString(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TString'))

    @classmethod
    def _from_java(cls, jtype):
        return TString()

    def convert(self, annotation):
        return annotation


class TArray(Type):
    def __init__(self, element_type):
        """
        :param :class:`.Type` element_type: Hail type of array element
        """
        jtype = scala_object(HailContext.hail_package().expr, 'TArray').apply(element_type._jtype)
        self.element_type = element_type
        super(jtype)

    @classmethod
    def _from_java(cls, jtype):
        t = TArray.__new__(cls)
        t.element_type = Type._from_java(jtype.elementType())
        t.super(jtype)
        return t

    def convert(self, annotation):
        return annotation


class TSet(Type):
    def __init__(self, element_type):
        """
        :param :class:`.Type` element_type: Hail type of set element
        """
        jtype = scala_object(HailContext.hail_package().expr, 'TSet').apply(element_type._jtype)
        self.element_type = element_type
        super(jtype)

    @classmethod
    def _from_java(cls, jtype):
        t = TSet.__new__(cls)
        t.element_type = Type._from_java(jtype.elementType())
        t.super(jtype)
        return t

    def convert(self, annotation):
        return annotation


class TDict(Type):
    def __init__(self, element_type):
        """
        :param :class:`.Type` element_type: Hail type of dict element
        """
        jtype = scala_object(HailContext.hail_package().expr, 'TDict').apply(element_type._jtype)
        self.element_type = element_type
        super(jtype)

    @classmethod
    def _from_java(cls, jtype):
        t = TDict.__new__(cls)
        t.element_type = Type._from_java(jtype.elementType())
        t.super(jtype)
        return t

    def convert(self, annotation):
        return annotation


class TStruct(Type):
    def __init__(self, names, types):
        """
        :param names: names of fields
        :type names: list of str
        :param types: types of fields
        :type types: list of :class:`.Type`
        """

        if len(names) != len(types):
            raise ValueError('length of names and types not equal: %d and %d' % (len(names), len(types)))
        jtype = scala_object(HailContext.hail_package().expr, 'TStruct').apply(names, types)
        self._init_from_java(jtype)


    @classmethod
    def _from_java(cls, jtype):
        struct = TStruct.__new__(cls)
        struct._init_from_java(jtype)
        return struct

    def _init_from_java(self, jtype):
        self.
    def convert(self, annotation):
        return annotation


class TVariant(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TVariant'))


class TAltAllele(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TAltAllele'))


class TGenotype(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TGenotype'))


class TLocus(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TLocus'))


class TInterval(Type):
    def __init__(self):
        super(scala_object(HailContext.hail_package().expr, 'TInterval'))
