package is.hail.expr

import is.hail.annotations.{Annotation, AnnotationPathException, _}
import is.hail.check.Arbitrary._
import is.hail.check.{Gen, _}
import is.hail.utils
import is.hail.utils.{Interval, StringEscapeUtils, _}
import is.hail.variant.{AltAllele, Call, Genotype, Locus, Variant}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DataType
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.reflect.classTag

object Type {
  val genScalar = Gen.oneOf[Type](TBoolean, TInt, TLong, TFloat, TDouble, TString,
    TVariant, TAltAllele, TGenotype, TLocus, TInterval, TCall)

  def genSized(size: Int): Gen[Type] = {
    if (size < 1)
      Gen.const(TStruct.empty)
    else if (size < 2)
      genScalar
    else
      Gen.oneOfGen(genScalar,
        genArb.resize(size - 1).map(TArray),
        genArb.resize(size - 1).map(TSet),
        Gen.zip(genArb, genArb).map { case (k, v) => TDict(k, v) },
        genStruct.resize(size))
  }

  def genStruct: Gen[TStruct] =
    Gen.buildableOf[Array, (String, Type, Map[String, String])](
      Gen.zip(Gen.identifier,
        genArb,
        Gen.option(
          Gen.buildableOf2[Map, String, String](
            Gen.zip(arbitrary[String].filter(s => !s.isEmpty), arbitrary[String])))
          .map(o => o.getOrElse(Map.empty[String, String]))))
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields => TStruct(fields
        .iterator
        .zipWithIndex
        .map { case ((k, t, m), i) => Field(k, t, i, m) }
        .toIndexedSeq))

  def genArb: Gen[Type] = Gen.sized(genSized)

  implicit def arbType = Arbitrary(genArb)

  def parseMap(s: String): Map[String, Type] = Parser.parseAnnotationTypes(s)
}

sealed abstract class Type {

  def children: Seq[Type] = Seq()

  def clear(): Unit = children.foreach(_.clear())

  def desc: String = ""

  def unify(concrete: Type): Boolean = {
    this == concrete
  }

  def isBound: Boolean = children.forall(_.isBound)

  def subst(): Type = this

  def getAsOption[T](fields: String*)(implicit ct: ClassTag[T]): Option[T] = {
    getOption(fields: _*)
      .flatMap { t =>
        if (ct.runtimeClass.isInstance(t))
          Some(t.asInstanceOf[T])
        else
          None
      }
  }

  def getOption(fields: String*): Option[Type] = getOption(fields.toList)

  def getOption(path: List[String]): Option[Type] = {
    if (path.isEmpty)
      Some(this)
    else
      None
  }

  def delete(fields: String*): (Type, Deleter) = delete(fields.toList)

  def delete(path: List[String]): (Type, Deleter) = {
    if (path.nonEmpty)
      throw new AnnotationPathException(s"invalid path ${ path.mkString(".") } from type ${ this }")
    else
      (TStruct.empty, a => Annotation.empty)
  }

  def insert(signature: Type, fields: String*): (Type, Inserter) = insert(signature, fields.toList)

  def insert(signature: Type, path: List[String]): (Type, Inserter) = {
    if (path.nonEmpty)
      TStruct.empty.insert(signature, path)
    else
      (signature, (a, toIns) => toIns)
  }

  def query(fields: String*): Querier = query(fields.toList)

  def query(path: List[String]): Querier = {
    if (path.nonEmpty)
      throw new AnnotationPathException(s"invalid path ${ path.mkString(".") } from type ${ this }")
    else
      identity[Annotation]
  }

  def toPrettyString(indent: Int = 0, compact: Boolean = false, printAttrs: Boolean = false): String = {
    val sb = new StringBuilder
    pretty(sb, indent, compact = compact, printAttrs = printAttrs)
    sb.result()
  }

  def pretty(sb: StringBuilder, indent: Int = 0, printAttrs: Boolean = false, compact: Boolean = false) {
    sb.append(toString)
  }

  def fieldOption(fields: String*): Option[Field] = fieldOption(fields.toList)

  def fieldOption(path: List[String]): Option[Field] =
    None

  def schema: DataType = SparkAnnotationImpex.exportType(this)

  def str(a: Annotation): String = if (a == null) "NA" else a.toString

  def toJSON(a: Annotation): JValue = JSONAnnotationImpex.exportAnnotation(a, this)

  def genNonmissingValue: Gen[Annotation] = Gen.const(Annotation.empty)

  def genValue: Gen[Annotation] = Gen.oneOfGen(Gen.const(Annotation.empty), genNonmissingValue)

  def isRealizable: Boolean = children.forall(_.isRealizable)

  def typeCheck(a: Any): Boolean

  /* compare values for equality, but compare Float and Double values using D_== */
  def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double = utils.defaultTolerance): Boolean = a1 == a2

  def scalaClassTag: ClassTag[_ <: AnyRef]

  def canCompare(other: Type): Boolean = this == other

  def ordering(missingGreatest: Boolean): Ordering[Annotation]
}

case object TBinary extends Type {
  override def toString = "Binary"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf(arbitrary[Byte])

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val ord = Ordering.Iterable[Byte]

    extendOrderingToNull(missingGreatest)(
      new Ordering[Array[Byte]] {
        def compare(a: Array[Byte], b: Array[Byte]): Int = ord.compare(a, b)
      })
  }
}

case object TBoolean extends Type {
  override def toString = "Boolean"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Boolean]

  def parse(s: String): Annotation = s.toBoolean

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Boolean]

  override def scalaClassTag: ClassTag[java.lang.Boolean] = classTag[java.lang.Boolean]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Boolean]])
}

object TNumeric {
  def promoteNumeric(types: Set[TNumeric]): Type = {
    if (types.size == 1)
      types.head
    else if (types(TDouble))
      TDouble
    else if (types(TFloat))
      TFloat
    else {
      assert(types(TLong))
      TLong
    }
  }
}

abstract class TNumeric extends Type {
  def conv: NumericConversion[_, _]

  override def canCompare(other: Type): Boolean = other.isInstanceOf[TNumeric]
}

abstract class TIntegral extends TNumeric

case object TInt extends TIntegral {
  override def toString = "Int"

  val conv = IntNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Int]

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Int]])

}

case object TLong extends TIntegral {
  override def toString = "Long"

  val conv = LongNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Long]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Long]

  override def scalaClassTag: ClassTag[java.lang.Long] = classTag[java.lang.Long]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Long]])
}

case object TFloat extends TNumeric {
  override def toString = "Float"

  val conv = FloatNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Float]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Float].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double].map(_.toFloat)

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null && D_==(a1.asInstanceOf[Float], a2.asInstanceOf[Float], tolerance))

  override def scalaClassTag: ClassTag[java.lang.Float] = classTag[java.lang.Float]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Float]])
}

case object TDouble extends TNumeric {
  override def toString = "Double"

  val conv = DoubleNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Double]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Double].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double]

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null && D_==(a1.asInstanceOf[Double], a2.asInstanceOf[Double], tolerance))

  override def scalaClassTag: ClassTag[java.lang.Double] = classTag[java.lang.Double]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Double]])
}

case object TString extends Type {
  override def toString = "String"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[String]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[String]

  override def scalaClassTag: ClassTag[String] = classTag[String]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[String]])
}

case class TFunction(paramTypes: Seq[Type], returnType: Type) extends Type {
  override def toString = s"(${ paramTypes.mkString(",") }) => $returnType"

  override def isRealizable = false

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TFunction(cparamTypes, creturnType) =>
        paramTypes.length == cparamTypes.length &&
          (paramTypes, cparamTypes).zipped.forall { case (pt, cpt) =>
            pt.unify(cpt)
          } &&
          returnType.unify(creturnType)

      case _ => false
    }
  }

  override def subst() = TFunction(paramTypes.map(_.subst()), returnType.subst())

  def typeCheck(a: Any): Boolean =
    throw new RuntimeException("TFunction is not realizable")

  override def children: Seq[Type] = paramTypes :+ returnType

  override def scalaClassTag: ClassTag[AnyRef] = throw new RuntimeException("TFunction is not realizable")

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    throw new RuntimeException("TFunction is not realizable")
}

case class Box[T](var b: Option[T] = None) {
  def unify(t: T): Boolean = b match {
    case Some(bt) => t == bt
    case None =>
      b = Some(t)
      true
  }

  def clear() {
    b = None
  }

  def get: T = b.get
}

case class TAggregableVariable(elementType: Type, st: Box[SymbolTable]) extends Type {
  override def toString = s"?Aggregable[$elementType]"

  override def isRealizable = false

  override def children = Seq(elementType)

  def typeCheck(a: Any): Boolean =
    throw new RuntimeException("TAggregableVariable is not realizable")

  override def unify(concrete: Type): Boolean = concrete match {
    case cagg: TAggregable =>
      elementType.unify(cagg.elementType) && st.unify(cagg.symTab)
    case _ => false
  }

  override def isBound: Boolean = elementType.isBound & st.b.nonEmpty

  override def clear() {
    st.clear()
  }

  override def subst(): Type = {
    assert(st != null)
    TAggregable(elementType.subst(), st.get)
  }

  override def desc: String = TAggregable.desc

  override def canCompare(other: Type): Boolean = false

  override def scalaClassTag: ClassTag[AnyRef] = throw new RuntimeException("TAggregableVariable is not realizable")

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    throw new RuntimeException("TAggregableVariable is not realizable")
}

case class TVariable(name: String, var t: Type = null) extends Type {
  override def toString: String = s"?$name"

  override def isRealizable = false

  def typeCheck(a: Any): Boolean =
    throw new RuntimeException("TVariable is not realizable")

  override def unify(concrete: Type): Boolean = {
    if (t == null) {
      t = concrete
      true
    } else
      t == concrete
  }

  override def isBound: Boolean = t != null

  override def clear() {
    t = null
  }

  override def subst(): Type = {
    assert(t != null)
    t
  }

  override def scalaClassTag: ClassTag[AnyRef] = throw new RuntimeException("TVariable is not realizable")

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    throw new RuntimeException("TVariable is not realizable")
}

object TAggregable {
  val desc = """An ``Aggregable`` is a Hail data type representing a distributed row or column of a matrix. Hail exposes a number of methods to compute on aggregables depending on the data type."""

  def apply(elementType: Type, symTab: SymbolTable): TAggregable = {
    val agg = TAggregable(elementType)
    agg.symTab = symTab
    agg
  }
}

case class TAggregable(elementType: Type) extends TContainer {
  // FIXME does symTab belong here?
  // not used for equality
  var symTab: SymbolTable = _

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TAggregable(celementType) => elementType.unify(celementType)
      case _ => false
    }
  }

  // FIXME symTab == null
  override def subst() = TAggregable(elementType.subst())

  override def isRealizable = false

  def typeCheck(a: Any): Boolean =
    throw new RuntimeException("TAggregable is not realizable")

  override def toString: String = s"Aggregable[${ elementType.toString }]"

  override def desc: String = TAggregable.desc

  override def scalaClassTag: ClassTag[_ <: AnyRef] = elementType.scalaClassTag

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    throw new RuntimeException("TAggregable is not realizable")
}

abstract class TContainer extends Type {
  def elementType: Type

  override def children = Seq(elementType)
}

abstract class TIterable extends TContainer {

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && (a1.asInstanceOf[Iterable[_]].size == a2.asInstanceOf[Iterable[_]].size)
      && a1.asInstanceOf[Iterable[_]].zip(a2.asInstanceOf[Iterable[_]])
      .forall { case (e1, e2) => elementType.valuesSimilar(e1, e2, tolerance) })

  override def canCompare(other: Type): Boolean = other match {
    case TArray(otherType) => elementType.canCompare(otherType)
    case _ => false
  }

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    extendOrderingToNull(missingGreatest)(
      Ordering.Iterable(elementType.ordering(missingGreatest)))
  }
}

case class TArray(elementType: Type) extends TIterable {
  override def toString = s"Array[$elementType]"

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TArray(celementType) => elementType.unify(celementType)
      case _ => false
    }
  }

  override def subst() = TArray(elementType.subst())

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean, compact: Boolean = false) {
    sb.append("Array[")
    elementType.pretty(sb, indent, printAttrs, compact)
    sb.append("]")
  }

  def typeCheck(a: Any): Boolean = a == null || (a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck))

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] =
    Gen.buildableOf[Array, Annotation](elementType.genValue).map(x => x: IndexedSeq[Annotation])


  override def desc: String =
    """
    An ``Array`` is a collection of items that all have the same data type (ex: Int, String) and are indexed. Arrays can be constructed by specifying ``[item1, item2, ...]`` and they are 0-indexed.

    An example of constructing an array and accessing an element is:

    .. code-block:: text
        :emphasize-lines: 2

        let a = [1, 10, 3, 7] in a[1]
        result: 10

    They can also be nested such as Array[Array[Int]]:

    .. code-block:: text
        :emphasize-lines: 2

        let a = [[1, 2, 3], [4, 5], [], [6, 7]] in a[1]
        result: [4, 5]
    """

  override def scalaClassTag: ClassTag[IndexedSeq[AnyRef]] = classTag[IndexedSeq[AnyRef]]
}

case class TSet(elementType: Type) extends TIterable {
  override def toString = s"Set[$elementType]"

  override def unify(concrete: Type): Boolean = concrete match {
    case TSet(celementType) => elementType.unify(celementType)
    case _ => false
  }

  override def subst() = TSet(elementType.subst())

  def typeCheck(a: Any): Boolean =
    a == null || (a.isInstanceOf[Set[_]] && a.asInstanceOf[Set[_]].forall(elementType.typeCheck))

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean, compact: Boolean = false) {
    sb.append("Set[")
    elementType.pretty(sb, indent, printAttrs, compact)
    sb.append("]")
  }

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf[Set, Annotation](elementType.genValue)

  override def desc: String =
    """
    A ``Set`` is an unordered collection with no repeated values of a given data type (ex: Int, String). Sets can be constructed by specifying ``[item1, item2, ...].toSet()``.

    .. code-block:: text
        :emphasize-lines: 2

        let s = ["rabbit", "cat", "dog", "dog"].toSet()
        result: Set("cat", "dog", "rabbit")

    They can also be nested such as Set[Set[Int]]:

    .. code-block:: text
        :emphasize-lines: 2

        let s = [[1, 2, 3].toSet(), [4, 5, 5].toSet()].toSet()
        result: Set(Set(1, 2, 3), Set(4, 5))
    """

  override def scalaClassTag: ClassTag[Set[AnyRef]] = classTag[Set[AnyRef]]
}

case class TDict(keyType: Type, valueType: Type) extends TContainer {

  override def canCompare(other: Type): Boolean = other match {
    case TDict(okt, ovt) => keyType.canCompare(okt) && valueType.canCompare(ovt)
    case _ => false
  }

  def elementType: Type = valueType

  override def children = Seq(keyType, valueType)

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TDict(kt, vt) => keyType.unify(kt) && valueType.unify(vt)
      case _ => false
    }
  }

  override def subst() = TDict(keyType.subst(), valueType.subst())

  override def toString = s"Dict[$keyType, $valueType]"

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean, compact: Boolean = false) {
    sb.append("Dict[")
    keyType.pretty(sb, indent, printAttrs, compact)
    if (compact)
      sb += ','
    else
      sb.append(", ")
    valueType.pretty(sb, indent, printAttrs, compact)
    sb.append("]")
  }

  def typeCheck(a: Any): Boolean = a == null || (a.isInstanceOf[Map[_, _]] &&
    a.asInstanceOf[Map[_, _]].forall { case (k, v) => keyType.typeCheck(k) && valueType.typeCheck(v) })

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] =
    Gen.buildableOf2[Map, Annotation, Annotation](Gen.zip(keyType.genValue, valueType.genValue))

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null) ||
      a1.asInstanceOf[Map[Any, _]].outerJoin(a2.asInstanceOf[Map[Any, _]])
        .forall { case (_, (o1, o2)) =>
          o1.liftedZip(o2).exists { case (v1, v2) => valueType.valuesSimilar(v1, v2, tolerance) }
        }

  override def desc: String =
    """
    A ``Dict`` is an unordered collection of key-value pairs. Each key can only appear once in the collection.
    """

  override def scalaClassTag: ClassTag[Map[_, _]] = classTag[Map[_, _]]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    extendOrderingToNull(missingGreatest)(
      Ordering.Iterable(
        Ordering.Tuple2(
          elementType.ordering(missingGreatest),
          elementType.ordering(missingGreatest))))
  }
}

case object TGenotype extends Type {
  override def toString = "Genotype"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Genotype]

  override def genNonmissingValue: Gen[Annotation] = Genotype.genArb

  override def desc: String = "A ``Genotype`` is a Hail data type representing a genotype in the Variant Dataset. It is referred to as ``g`` in the expression language."

  override def scalaClassTag: ClassTag[Genotype] = classTag[Genotype]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Genotype]])
}

case object TCall extends Type {
  override def toString = "Call"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = Call.genArb

  override def desc: String = "A ``Call`` is a Hail data type representing a genotype call (ex: 0/0) in the Variant Dataset."

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Int]])
}

case object TAltAllele extends Type {
  override def toString = "AltAllele"

  def typeCheck(a: Any): Boolean = a == null || a == null || a.isInstanceOf[AltAllele]

  override def genNonmissingValue: Gen[Annotation] = AltAllele.gen

  override def desc: String = "An ``AltAllele`` is a Hail data type representing an alternate allele in the Variant Dataset."

  override def scalaClassTag: ClassTag[AltAllele] = classTag[AltAllele]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[AltAllele]])
}

case object TVariant extends Type {
  override def toString = "Variant"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Variant]

  override def genNonmissingValue: Gen[Annotation] = Variant.gen

  override def desc: String =
    """
    A ``Variant`` is a Hail data type representing a variant in the Variant Dataset. It is referred to as ``v`` in the expression language.

    The `pseudoautosomal region <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`_ (PAR) is currently defined with respect to reference `GRCh37 <http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/>`_:

    - X: 60001 - 2699520, 154931044 - 155260560
    - Y: 10001 - 2649520, 59034050 - 59363566

    Most callers assign variants in PAR to X.
    """

  override def scalaClassTag: ClassTag[Variant] = classTag[Variant]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Variant]])
}

case object TLocus extends Type {
  override def toString = "Locus"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Locus]

  override def genNonmissingValue: Gen[Annotation] = Locus.gen

  override def desc: String = "A ``Locus`` is a Hail data type representing a specific genomic location in the Variant Dataset."

  override def scalaClassTag: ClassTag[Locus] = classTag[Locus]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Locus]])
}

case object TInterval extends Type {
  override def toString = "Interval"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Interval[_]] && a.asInstanceOf[Interval[_]].end.isInstanceOf[Locus]

  override def genNonmissingValue: Gen[Annotation] = Interval.gen(Locus.gen)

  override def desc: String = "An ``Interval`` is a Hail data type representing a range of genomic locations in the Variant Dataset."

  override def scalaClassTag: ClassTag[Interval[Locus]] = classTag[Interval[Locus]]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    extendOrderingToNull(missingGreatest)(implicitly[Ordering[Interval[Locus]]])
}

case class Field(name: String, typ: Type,
  index: Int,
  attrs: Map[String, String] = Map.empty) {
  def attr(s: String): Option[String] = attrs.get(s)

  def attrsJava(): java.util.Map[String, String] = attrs.asJava

  def unify(cf: Field): Boolean =
    name == cf.name &&
      typ.unify(cf.typ) &&
      index == cf.index &&
      attrs == cf.attrs

  def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean, compact: Boolean) {
    if (compact) {
      sb.append(prettyIdentifier(name))
      sb.append(":")
    } else {
      sb.append(" " * indent)
      sb.append(prettyIdentifier(name))
      sb.append(": ")
    }
    typ.pretty(sb, indent, printAttrs, compact)
    if (printAttrs) {
      attrs.foreach { case (k, v) =>
        if (!compact) {
          sb += '\n'
          sb.append(" " * (indent + 2))
        }
        sb += '@'
        sb.append(prettyIdentifier(k))
        sb.append("=\"")
        sb.append(StringEscapeUtils.escapeString(v))
        sb += '"'
      }
    }
  }
}

object TStruct {
  def empty: TStruct = TStruct(Array.empty[Field])

  def apply(args: (String, Type)*): TStruct =
    TStruct(args
      .iterator
      .zipWithIndex
      .map { case ((n, t), i) => Field(n, t, i) }
      .toArray)

  def apply(names: java.util.ArrayList[String], types: java.util.ArrayList[Type]): TStruct = {
    val sNames = names.asScala.toArray
    val sTypes = types.asScala.toArray
    if (sNames.length != sTypes.length)
      fatal(s"number of names does not match number of types: found ${ sNames.length } names and ${ sTypes.length } types")

    TStruct(sNames.zip(sTypes): _*)
  }
}

case class TStruct(fields: IndexedSeq[Field]) extends Type {
  override def children: Seq[Type] = fields.map(_.typ)

  override def canCompare(other: Type): Boolean = other match {
    case t: TStruct => size == t.size && fields.zip(t.fields).forall { case (f1, f2) =>
      f1.name == f2.name && f1.typ.canCompare(f2.typ)
    }
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TStruct(cfields) =>
      fields.length == cfields.length &&
        (fields, cfields).zipped.forall { case (f, cf) =>
          f.unify(cf)
        }
    case _ => false
  }

  override def subst() = TStruct(fields.map(f => f.copy(typ = f.typ.subst().asInstanceOf[Type])))

  val fieldIdx: Map[String, Int] =
    fields.map(f => (f.name, f.index)).toMap

  def index(str: String): Option[Int] = fieldIdx.get(str)

  def selfField(name: String): Option[Field] = fieldIdx.get(name).map(i => fields(i))

  def hasField(name: String): Boolean = fieldIdx.contains(name)

  def size: Int = fields.length

  override def getOption(path: List[String]): Option[Type] =
    if (path.isEmpty)
      Some(this)
    else
      selfField(path.head).map(_.typ).flatMap(t => t.getOption(path.tail))

  override def fieldOption(path: List[String]): Option[Field] =
    if (path.isEmpty)
      None
    else {
      val f = selfField(path.head)
      if (path.length == 1)
        f
      else
        f.flatMap(_.typ.fieldOption(path.tail))
    }

  def updateFieldAttributes(path: List[String], f: Map[String, String] => Map[String, String]): TStruct = {

    if (path.isEmpty)
      throw new AnnotationPathException(s"Empty path for attribute annotation is not allowed.")

    if (!hasField(path.head))
      throw new AnnotationPathException(s"struct has no field ${ path.head }")

    copy(fields.map {
      field =>
        if (field.name == path.head) {
          if (path.length == 1)
            field.copy(attrs = f(field.attrs))
          else {
            field.typ match {
              case struct: TStruct => field.copy(typ = struct.updateFieldAttributes(path.tail, f))
              case t => fatal(s"Field ${ field.name } is not a Struct and cannot contain field ${ path.tail.mkString(".") }")
            }
          }
        }
        else
          field
    })
  }

  def setFieldAttributes(path: List[String], kv: Map[String, String]): TStruct = {
    updateFieldAttributes(path, attributes => attributes ++ kv)
  }

  def deleteFieldAttribute(path: List[String], attr: String): TStruct = {
    updateFieldAttributes(path, attributes => attributes - attr)
  }

  override def query(p: List[String]): Querier = {
    if (p.isEmpty)
      identity[Annotation]
    else {
      selfField(p.head) match {
        case Some(f) =>
          val q = f.typ.query(p.tail)
          val localIndex = f.index
          a =>
            if (a == Annotation.empty)
              null
            else
              q(a.asInstanceOf[Row].get(localIndex))
        case None => throw new AnnotationPathException(s"struct has no field ${ p.head }")
      }
    }
  }

  override def delete(p: List[String]): (Type, Deleter) = {
    if (p.isEmpty)
      (TStruct.empty, a => Annotation.empty)
    else {
      val key = p.head
      val f = selfField(key) match {
        case Some(f) => f
        case None => throw new AnnotationPathException(s"$key not found")
      }
      val index = f.index
      val (newFieldType, d) = f.typ.delete(p.tail)
      val newType: Type =
        if (newFieldType == TStruct.empty)
          deleteKey(key, f.index)
        else
          updateKey(key, f.index, newFieldType)

      val localDeleteFromRow = newFieldType == TStruct.empty

      val deleter: Deleter = { a =>
        if (a == Annotation.empty)
          Annotation.empty
        else {
          val r = a.asInstanceOf[Row]

          if (localDeleteFromRow)
            r.delete(index)
          else
            r.update(index, d(r.get(index)))
        }
      }
      (newType, deleter)
    }
  }

  override def insert(signature: Type, p: List[String]): (Type, Inserter) = {
    if (p.isEmpty)
      (signature, (a, toIns) => toIns)
    else {
      val key = p.head
      val f = selfField(key)
      val keyIndex = f.map(_.index)
      val (newKeyType, keyF) = f
        .map(_.typ)
        .getOrElse(TStruct.empty)
        .insert(signature, p.tail)

      val newSignature = keyIndex match {
        case Some(i) => updateKey(key, i, newKeyType)
        case None => appendKey(key, newKeyType)
      }

      val localSize = fields.size

      val inserter: Inserter = (a, toIns) => {
        val r = if (a == null || localSize == 0) // localsize == 0 catches cases where we overwrite a path
          Row.fromSeq(Array.fill[Any](localSize)(null))
        else
          a.asInstanceOf[Row]
        keyIndex match {
          case Some(i) => r.update(i, keyF(r.get(i), toIns))
          case None => r.append(keyF(Annotation.empty, toIns))
        }
      }
      (newSignature, inserter)
    }
  }

  def updateKey(key: String, i: Int, sig: Type): Type = {
    assert(fieldIdx.contains(key))

    val newFields = Array.fill[Field](fields.length)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(i) = Field(key, sig, i)
    TStruct(newFields)
  }

  def deleteKey(key: String, index: Int): Type = {
    assert(fieldIdx.contains(key))
    if (fields.length == 1)
      TStruct.empty
    else {
      val newFields = Array.fill[Field](fields.length - 1)(null)
      for (i <- 0 until index)
        newFields(i) = fields(i)
      for (i <- index + 1 until fields.length)
        newFields(i - 1) = fields(i).copy(index = i - 1)
      TStruct(newFields)
    }
  }

  def appendKey(key: String, sig: Type): TStruct = {
    assert(!fieldIdx.contains(key))
    val newFields = Array.fill[Field](fields.length + 1)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(fields.length) = Field(key, sig, fields.length)
    TStruct(newFields)
  }

  def merge(other: TStruct): (TStruct, Merger) = {

    val intersect = fields.map(_.name).toSet
      .intersect(other.fields.map(_.name).toSet)

    if (intersect.nonEmpty)
      fatal(
        s"""Invalid merge operation: cannot merge structs with same-name ${ plural(intersect.size, "field") }
           |  Found these fields in both structs: [ ${
          intersect.map(s => prettyIdentifier(s)).mkString(", ")
        } ]
           |  Hint: use `drop' or `select' to remove these fields from one side""".stripMargin)

    val newStruct = TStruct(fields ++ other.fields.map(f => f.copy(index = f.index + size)))

    val size1 = size
    val size2 = other.size
    val targetSize = newStruct.size

    val merger = (a1: Annotation, a2: Annotation) => {
      if (a1 == null && a2 == null)
        Annotation.empty
      else {
        val s1 = Option(a1).map(_.asInstanceOf[Row].toSeq)
          .getOrElse(Seq.fill[Any](size1)(null))
        val s2 = Option(a2).map(_.asInstanceOf[Row].toSeq)
          .getOrElse(Seq.fill[Any](size2)(null))
        val newValues = s1 ++ s2
        assert(newValues.size == targetSize)
        Annotation.fromSeq(newValues)
      }
    }

    (newStruct, merger)
  }

  def filter(set: Set[String], include: Boolean = true): (TStruct, Deleter) = {
    val notFound = set.filter(name => selfField(name).isEmpty).map(prettyIdentifier)
    if (notFound.nonEmpty)
      fatal(
        s"""invalid struct filter operation: ${
          plural(notFound.size, s"field ${ notFound.head }", s"fields [ ${ notFound.mkString(", ") } ]")
        } not found
           |  Existing struct fields: [ ${ fields.map(f => prettyIdentifier(f.name)).mkString(", ") } ]""".stripMargin)

    val fn = (f: Field) =>
      if (include)
        set.contains(f.name)
      else
        !set.contains(f.name)
    filter(fn)
  }

  def parseInStructScope[T >: Null](code: String)(implicit hr: HailRep[T]): (Annotation) => T = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val f = Parser.parseTypedExpr[T](code, ec)

    (a: Annotation) => {
      if (a == null)
        null
      else {
        ec.setAllFromRow(a.asInstanceOf[Row])
        f()
      }
    }
  }

  def filter(f: (Field) => Boolean): (TStruct, Deleter) = {
    val included = fields.map(f)

    val newFields = fields.zip(included)
      .flatMap { case (field, incl) =>
        if (incl)
          Some(field)
        else
          None
      }

    val newSize = newFields.size

    val filterer = (a: Annotation) =>
      if (a == null)
        a
      else if (newSize == 0)
        Annotation.empty
      else {
        val r = a.asInstanceOf[Row]
        val newValues = included.zipWithIndex
          .flatMap {
            case (incl, i) =>
              if (incl)
                Some(r.get(i))
              else None
          }
        assert(newValues.length == newSize)
        Annotation.fromSeq(newValues)
      }

    (TStruct(newFields.zipWithIndex.map { case (f, i) => f.copy(index = i) }), filterer)
  }

  override def toString: String = if (size == 0) "Empty" else toPrettyString(compact = true)

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean, compact: Boolean) {
    if (size == 0)
      sb.append("Empty")
    else {
      if (compact) {
        sb.append("Struct{")
        fields.foreachBetween(_.pretty(sb, indent, printAttrs, compact))(sb += ',')
        sb += '}'
      } else {
        sb.append("Struct{")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, printAttrs, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  override def typeCheck(a: Any): Boolean =
    a == null ||
      (a.isInstanceOf[Row] && {
        val r = a.asInstanceOf[Row]
        r.length == fields.length &&
          r.toSeq.zip(fields).forall {
            case (v, f) => f.typ.typeCheck(v)
          }
      })

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] = {
    if (size == 0)
      Gen.const(Annotation.empty)
    else
      Gen.size.flatMap(fuel =>
        if (size < fuel) Gen.const(Annotation.empty)
        else Gen.uniformSequence(fields.map(f => f.typ.genValue)).map(a => Annotation(a: _*)))
  }

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && fields.zip(a1.asInstanceOf[Row].toSeq).zip(a2.asInstanceOf[Row].toSeq)
      .forall {
        case ((f, x1), x2) =>
          f.typ.valuesSimilar(x1, x2, tolerance)
      })

  override def desc: String =
    """
    A ``Struct`` is like a Python tuple where the fields are named and the set of fields is fixed.

    An example of constructing and accessing the fields in a ``Struct`` is

    .. code-block:: text
        :emphasize-lines: 2

        let s = {gene: "ACBD", function: "LOF", nHet: 12} in s.gene
        result: "ACBD"

    A field of the ``Struct`` can also be another ``Struct``. For example, ``va.info.AC`` selects the struct ``info`` from the struct ``va``, and then selects the array ``AC`` from the struct ``info``.
    """

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val fieldOrderings = fields.map(f => f.typ.ordering(missingGreatest))

    extendOrderingToNull(missingGreatest)(new Ordering[Row] {
      def compare(a: Row, b: Row): Int = {
        var i = 0
        while (i < a.size) {
          val c = fieldOrderings(i).compare(a.get(i), b.get(i))
          if (c != 0)
            return c

          i += 1
        }

        // equal
        0
      }
    })
  }
}
