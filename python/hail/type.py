from hail.java import scala_object, scala_package_object, Env
from hail.representation import Variant, AltAllele, Genotype, Locus, Interval, Struct


class TypeCheckError(Exception):
    def __init__(self, message):
        self.msg = message
        super(TypeCheckError).__init__(TypeCheckError)

    def __str__(self):
        return self.msg


class Type(object):
    """
    Hail type class used for annotations and expression language.

    :param jtype: equivalent java type
    """

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
        # FIXME this is pretty hacky
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
    Hail type corresponding to int
    """

    def __init__(self):
        super(TInt, self).__init__(scala_object(Env.hail_package().expr, 'TInt'))

    def convert_to_py(self, annotation):
        return annotation

    def convert_to_j(self, annotation):
        if annotation:
            return scala_package_object(Env.hail_package().utils).makeInt(annotation)
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, int):
            raise TypeCheckError("TInt expected type 'int', but found type '%s'" % type(annotation))


class TLong(Type):
    """
    Hail type corresponding to long
    """

    def __init__(self):
        super(TLong, self).__init__(scala_object(Env.hail_package().expr, 'TLong'))

    def convert_to_py(self, annotation):
        return annotation

    def convert_to_j(self, annotation):
        if annotation:
            return scala_package_object(Env.hail_package().utils).makeLong(annotation)
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not (isinstance(annotation, long) or isinstance(annotation, int)):
            raise TypeCheckError("TLong expected type 'int' or 'long', but found type '%s'" % type(annotation))


class TFloat(Type):
    """
    Hail type corresponding to float
    """

    def __init__(self):
        super(TFloat, self).__init__(scala_object(Env.hail_package().expr, 'TFloat'))

    def convert_to_py(self, annotation):
        return annotation

    def convert_to_j(self, annotation):
        # if annotation:
        #     return scala_package_object(Env.hail_package().utils).makeFloat(annotation)
        # else:
        #     return annotation

        # FIXME: This function is unsupported until py4j-0.10.4: https://github.com/bartdag/py4j/issues/255
        raise NotImplementedError('TFloat is currently unsupported in certain operations, use TDouble instead')

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, float):
            raise TypeCheckError("TDouble expected type 'float', but found type '%s'" % type(annotation))


class TDouble(Type):
    """
    Hail type corresponding to float
    """

    def __init__(self):
        super(TDouble, self).__init__(scala_object(Env.hail_package().expr, 'TDouble'))

    def convert_to_py(self, annotation):
        return annotation

    def convert_to_j(self, annotation):
        if annotation:
            return scala_package_object(Env.hail_package().utils).makeDouble(annotation)
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, float):
            raise TypeCheckError("TDouble expected type 'float', but found type '%s'" % type(annotation))


class TString(Type):
    """
    Hail type corresponding to str
    """

    def __init__(self):
        super(TString, self).__init__(scala_object(Env.hail_package().expr, 'TString'))

    def convert_to_py(self, annotation):
        return annotation

    def convert_to_j(self, annotation):
        return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, str):
            raise TypeCheckError("TString expected type 'str', but found type '%s'" % type(annotation))


class TBoolean(Type):
    """
    Hail type corresponding to bool
    """

    def __init__(self):
        super(TBoolean, self).__init__(scala_object(Env.hail_package().expr, 'TBoolean'))

    def convert_to_py(self, annotation):
        return annotation

    def convert_to_j(self, annotation):
        return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, bool):
            raise TypeCheckError("TBoolean expected type 'bool', but found type '%s'" % type(annotation))


class TArray(Type):
    """
    Hail type corresponding to list

    :param element_type: type of array elements
    :type element_type: :class:`.Type`

    :ivar element_type: type of array elements
    :vartype element_type: :class:`.Type`
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

    def convert_to_py(self, annotation):
        if annotation:
            lst = scala_package_object(Env.hail_package().utils).iterableToArrayList(annotation)
            return [self.element_type.convert_to_py(x) for x in lst]
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return scala_package_object(Env.hail_package().utils).arrayListToISeq(
                [self.element_type.convert_to_j(elt) for elt in annotation]
            )
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation:
            if not isinstance(annotation, list):
                raise TypeCheckError("TArray expected type 'list', but found type '%s'" % type(annotation))
            for elt in annotation:
                self.element_type.typecheck(elt)


class TSet(Type):
    """
    Hail type corresponding to set

    :param element_type: type of set elements
    :type element_type: :class:`.Type`

    :ivar element_type: type of set elements
    :vartype element_type: :class:`.Type`
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

    def convert_to_py(self, annotation):
        if annotation:
            lst = scala_package_object(Env.hail_package().utils).iterableToArrayList(annotation)
            return set([self.element_type.convert_to_py(x) for x in lst])
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return scala_package_object(Env.hail_package().utils).arrayListToSet(
                [self.element_type.convert_to_j(elt) for elt in annotation]
            )
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation:
            if not isinstance(annotation, set):
                raise TypeCheckError("TSet expected type 'set', but found type '%s'" % type(annotation))
            for elt in annotation:
                self.element_type.typecheck(elt)


class TDict(Type):
    """
    Hail type corresponding to dict

    :param element_type: type of dict values
    :type element_type: :class:`.Type`

    :ivar element_type: type of dict values
    :vartype element_type: :class:`.Type`
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

    def convert_to_py(self, annotation):
        if annotation:
            lst = scala_package_object(Env.hail_package().utils).iterableToArrayList(annotation)
            d = dict()
            for x in lst:
                d[x._1()] = self.element_type.convert_to_py(x._2())
            return d
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return scala_package_object(Env.hail_package().utils).javaMapToMap(
                {k: self.element_type.convert_to_j(v) for k, v in annotation.iteritems()}
            )
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation:
            if not isinstance(annotation, dict):
                raise TypeCheckError("TDict expected type 'dict', but found type '%s'" % type(annotation))
            for v in annotation.values():
                self.element_type.typecheck(v)


class Field(object):
    def __init__(self, name, htype):
        self.name = name
        self.typ = htype


class TStruct(Type):
    """
    Hail type corresponding to :class:`hail.representation.Struct`

    :param names: names of fields
    :type names: list of str
    :param types: types of fields
    :type types: list of :class:`.Type`

    :ivar fields: struct fields
    :vartype fields: list of :class:`.Field`
    """

    def __init__(self, names, types):
        """
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

    def convert_to_py(self, annotation):
        if annotation:
            d = dict()
            for i, f in enumerate(self.fields):
                d[f.name] = f.typ.convert_to_py(annotation.get(i))
            return Struct(d, [f.name for f in self.fields])
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return scala_object(Env.hail_package().annotations, 'Annotation').fromSeq(
                scala_package_object(Env.hail_package().utils).arrayListToISeq(
                    [f.typ.convert_to_j(annotation[f.name]) for f in self.fields]
                )
            )
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation:
            for f in self.fields:
                if not (f.name in annotation):
                    raise TypeCheckError("TStruct expected fields '%s', but found fields '%s'" %
                                         ([f.name for f in self.fields], annotation.fields))
                f.typ.typecheck((annotation[f.name]))


class TVariant(Type):
    """
    Hail type corresponding to :class:`hail.representation.Variant`
    """

    def __init__(self):
        super(TVariant, self).__init__(scala_object(Env.hail_package().expr, 'TVariant'))

    def convert_to_py(self, annotation):
        if annotation:
            return Variant._from_java(annotation)
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, Variant):
            raise TypeCheckError('TVariant expected type hail.representation.Variant, but found %s' %
                                 type(annotation))


class TAltAllele(Type):
    """
    Hail type corresponding to :class:`hail.representation.AltAllele`
    """

    def __init__(self):
        super(TAltAllele, self).__init__(scala_object(Env.hail_package().expr, 'TAltAllele'))

    def convert_to_py(self, annotation):
        if annotation:
            return AltAllele._from_java(annotation)
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, AltAllele):
            raise TypeCheckError('TAltAllele expected type hail.representation.AltAllele, but found %s' %
                                 type(annotation))


class TGenotype(Type):
    """
    Hail type corresponding to :class:`hail.representation.Genotype`
    """

    def __init__(self):
        super(TGenotype, self).__init__(scala_object(Env.hail_package().expr, 'TGenotype'))

    def convert_to_py(self, annotation):
        if annotation:
            return Genotype._from_java(annotation)
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, Genotype):
            raise TypeCheckError('TGenotype expected type hail.representation.Genotype, but found %s' %
                                 type(annotation))


class TLocus(Type):
    """
    Hail type corresponding to :class:`hail.representation.Locus`
    """

    def __init__(self):
        super(TLocus, self).__init__(scala_object(Env.hail_package().expr, 'TLocus'))

    def convert_to_py(self, annotation):
        if annotation:
            return Locus._from_java(annotation)
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, Locus):
            raise TypeCheckError('TLocus expected type hail.representation.Locus, but found %s' %
                                 type(annotation))


class TInterval(Type):
    """
    Hail type corresponding to :class:`hail.representation.Interval`
    """

    def __init__(self):
        super(TInterval, self).__init__(scala_object(Env.hail_package().expr, 'TInterval'))

    def convert_to_py(self, annotation):
        if annotation:
            return Interval._from_java(annotation)
        else:
            return annotation

    def convert_to_j(self, annotation):
        if annotation is not None:
            return annotation._jrep
        else:
            return annotation

    def typecheck(self, annotation):
        if annotation and not isinstance(annotation, Interval):
            raise TypeCheckError('TInterval expected type hail.representation.Interval, but found %s' %
                                 type(annotation))
