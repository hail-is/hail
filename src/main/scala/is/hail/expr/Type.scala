package is.hail.expr

import is.hail.annotations.{Annotation, AnnotationPathException, _}
import is.hail.check.Arbitrary._
import is.hail.check.{Gen, _}
import is.hail.sparkextras.OrderedKey
import is.hail.utils
import is.hail.utils.{Interval, StringEscapeUtils, _}
import is.hail.variant.{AltAllele, Call, Contig, GenomeReference, Genotype, Locus, Variant}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DataType
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.reflect.classTag

object Type {
  val genScalar: Gen[Type] = Gen.oneOf[Type](TBoolean, TInt32, TInt64, TFloat32, TFloat64, TString,
    TVariant(GenomeReference.GRCh37), TAltAllele, TGenotype, TLocus(GenomeReference.GRCh37), TInterval(GenomeReference.GRCh37), TCall)

  def genSized(size: Int): Gen[Type] = {
    if (size < 1)
      Gen.const(TStruct.empty)
    else if (size < 2)
      genScalar
    else
      Gen.oneOfGen(genScalar,
        genScalar,
        genScalar,
        genScalar,
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
            Gen.zip(arbitrary[String].filter(s => !s.isEmpty), arbitrary[String])), someFraction = 0.05)
          .map(o => o.getOrElse(Map.empty[String, String]))))
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields => TStruct(fields
        .iterator
        .zipWithIndex
        .map { case ((k, t, m), i) => Field(k, t, i, m) }
        .toIndexedSeq))

  def genArb: Gen[Type] = Gen.sized(genSized)

  def genWithValue: Gen[(Type, Annotation)] = for {
    s <- Gen.size
    // prefer smaller type and bigger values
    fraction <- Gen.choose(0.1, 0.3)
    x = (fraction * s).toInt
    y = s - x
    t <- Type.genStruct.resize(x)
    v <- t.genValue.resize(y)
  } yield (t, v)

  implicit def arbType = Arbitrary(genArb)

  def parseMap(s: String): Map[String, Type] = Parser.parseAnnotationTypes(s)
}

sealed abstract class Type extends Serializable { self =>

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

  def unsafeOrdering(missingGreatest: Boolean = false): UnsafeOrdering = ???

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
      (TStruct.empty, a => null)
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
    val (t, q) = queryTyped(path)
    q
  }

  def queryTyped(fields: String*): (Type, Querier) = queryTyped(fields.toList)

  def queryTyped(path: List[String]): (Type, Querier) = {
    if (path.nonEmpty)
      throw new AnnotationPathException(s"invalid path ${ path.mkString(".") } from type ${ this }")
    else
      (this, identity[Annotation])
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

  def genNonmissingValue: Gen[Annotation] = ???

  def genValue: Gen[Annotation] = Gen.nextCoin(0.05)
    .flatMap(isEmpty => if (isEmpty) Gen.const(null) else genNonmissingValue)

  def isRealizable: Boolean = children.forall(_.isRealizable)

  def typeCheck(a: Any): Boolean

  /* compare values for equality, but compare Float and Double values using D_== */
  def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double = utils.defaultTolerance): Boolean = a1 == a2

  def scalaClassTag: ClassTag[_ <: AnyRef]

  def canCompare(other: Type): Boolean = this == other

  def ordering(missingGreatest: Boolean): Ordering[Annotation]

  val partitionKey: Type = this

  def orderedKey: OrderedKey[Annotation, Annotation] = new OrderedKey[Annotation, Annotation] {
    def project(key: Annotation): Annotation = key

    val kOrd: Ordering[Annotation] = ordering(missingGreatest = true)

    val pkOrd: Ordering[Annotation] = ordering(missingGreatest = true)

    val kct: ClassTag[Annotation] = classTag[Annotation]

    val pkct: ClassTag[Annotation] = classTag[Annotation]
  }

  def jsonReader: JSONReader[Annotation] = new JSONReader[Annotation] {
    def fromJSON(a: JValue): Annotation = JSONAnnotationImpex.importAnnotation(a, self)
  }

  def jsonWriter: JSONWriter[Annotation] = new JSONWriter[Annotation] {
    def toJSON(pk: Annotation): JValue = JSONAnnotationImpex.exportAnnotation(pk, self)
  }

  def byteSize: Long = throw new NotImplementedError(toString)

  def alignment: Long = byteSize

  /*  Fundamental types are types that can be handled natively by RegionValueBuilder: primitive
      types, Array and Struct. */
  def fundamentalType: Type = this
}

abstract class ComplexType extends Type {
  def representation: Type

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = representation.unsafeOrdering(missingGreatest)

  override lazy val fundamentalType: Type = representation.fundamentalType
}

case object TBinary extends Type {
  override def toString = "Binary"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf(arbitrary[Byte])

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
      val l1 = loadLength(r1, o1)
      val l2 = loadLength(r2, o2)

      val bOff1 = bytesOffset(o1)
      val bOff2 = bytesOffset(o2)

      val lim = math.min(l1, l2)
      var i = 0

      while (i < lim) {
        val b1 = r1.loadByte(bOff1 + i)
        val b2 = r2.loadByte(bOff2 + i)
        if (b1 != b2)
          return java.lang.Byte.compare(b1, b2)

        i += 1
      }
      Integer.compare(l1, l2)
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val ord = Ordering.Iterable[Byte]

    annotationOrdering(extendOrderingToNull(missingGreatest)(
      new Ordering[Array[Byte]] {
        def compare(a: Array[Byte], b: Array[Byte]): Int = ord.compare(a, b)
      }))
  }

  override def byteSize: Long = 8

  def contentAlignment: Long = 4

  def contentByteSize(length: Int): Int = 4 + length

  def loadLength(region: MemoryBuffer, boff: Long): Int =
    region.loadInt(boff)

  def bytesOffset(boff: Long): Long = boff + 4

  def allocate(region: MemoryBuffer, length: Int): Long = {
    region.align(contentAlignment)
    region.allocate(contentByteSize(length))
  }
}

case object TBoolean extends Type {
  override def toString = "Boolean"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Boolean]

  def parse(s: String): Annotation = s.toBoolean

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Boolean]

  override def scalaClassTag: ClassTag[java.lang.Boolean] = classTag[java.lang.Boolean]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
      java.lang.Boolean.compare(r1.loadBoolean(o1), r2.loadBoolean(o1))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Boolean]]))

  override def byteSize: Long = 1
}

object TNumeric {
  def promoteNumeric(types: Set[TNumeric]): Type = {
    if (types.size == 1)
      types.head
    else if (types(TFloat64))
      TFloat64
    else if (types(TFloat32))
      TFloat32
    else {
      assert(types(TInt64))
      TInt64
    }
  }
}

abstract class TNumeric extends Type {
  def conv: NumericConversion[_, _]

  override def canCompare(other: Type): Boolean = other.isInstanceOf[TNumeric]
}

abstract class TIntegral extends TNumeric

case object TInt32 extends TIntegral {
  override def toString = "Int32"

  val conv = IntNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Int]

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
      Integer.compare(r1.loadInt(o1), r2.loadInt(o2))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Int]]))

  override def byteSize: Long = 4
}

case object TInt64 extends TIntegral {
  override def toString = "Int64"

  val conv = LongNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Long]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Long]

  override def scalaClassTag: ClassTag[java.lang.Long] = classTag[java.lang.Long]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
      java.lang.Long.compare(r1.loadLong(o1), r2.loadLong(o2))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Long]]))

  override def byteSize: Long = 8
}

case object TFloat32 extends TNumeric {
  override def toString = "Float32"

  val conv = FloatNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Float]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Float].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double].map(_.toFloat)

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null &&
      (D_==(a1.asInstanceOf[Float], a2.asInstanceOf[Float], tolerance) ||
        (a1.asInstanceOf[Double].isNaN && a2.asInstanceOf[Double].isNaN)))

  override def scalaClassTag: ClassTag[java.lang.Float] = classTag[java.lang.Float]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
      java.lang.Float.compare(r1.loadFloat(o1), r2.loadFloat(o2))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Float]]))

  override def byteSize: Long = 4
}

case object TFloat64 extends TNumeric {
  override def toString = "Float64"

  val conv = DoubleNumericConversion

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Double]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Double].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double]

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null &&
      (D_==(a1.asInstanceOf[Double], a2.asInstanceOf[Double], tolerance) ||
        (a1.asInstanceOf[Double].isNaN && a2.asInstanceOf[Double].isNaN)))

  override def scalaClassTag: ClassTag[java.lang.Double] = classTag[java.lang.Double]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
      java.lang.Double.compare(r1.loadDouble(o1), r2.loadDouble(o2))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Double]]))

  override def byteSize: Long = 8
}

case object TString extends Type {
  override def toString = "String"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[String]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[String]

  override def scalaClassTag: ClassTag[String] = classTag[String]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = TBinary.unsafeOrdering(missingGreatest)

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[String]]))

  override def byteSize: Long = 8

  override lazy val fundamentalType: Type = TBinary

  def loadString(region: MemoryBuffer, offset: Long): String = {
    val length = TBinary.loadLength(region, offset)
    new String(region.loadBytes(TBinary.bytesOffset(offset), length))
  }
}

final case class TFunction(paramTypes: Seq[Type], returnType: Type) extends Type {
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

final case class Box[T](var b: Option[T] = None) {
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

final case class TAggregableVariable(elementType: Type, st: Box[SymbolTable]) extends Type {
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

final case class TVariable(name: String, var t: Type = null) extends Type {
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

final case class TAggregable(elementType: Type) extends TContainer {
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

  override def byteSize: Long = 8

  override def children = Seq(elementType)

  // FIXME LCM
  def contentsAlignment: Long = elementType.alignment.max(4)

  def elementsOffset(length: Int): Long =
    UnsafeUtils.roundUpAlignment(4 + (length + 7) / 8, elementType.alignment)

  def elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  def contentsByteSize(length: Int): Long =
    elementsOffset(length) + length * elementByteSize

  def loadLength(region: MemoryBuffer, aoff: Long): Int =
    region.loadInt(aoff)

  def isElementDefined(region: MemoryBuffer, aoff: Long, i: Int): Boolean =
    !region.loadBit(aoff + 4, i)

  def setElementMissing(region: MemoryBuffer, aoff: Long, i: Int) {
    region.setBit(aoff + 4, i)
  }

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    aoff + elementsOffset(length) + i * UnsafeUtils.arrayElementSize(elementType)

  def elementOffset(region: MemoryBuffer, aoff: Long, i: Int): Long =
    elementOffset(aoff, region.loadInt(aoff), i)

  def loadElement(region: MemoryBuffer, aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: TArray | TBinary => region.loadInt(off)
      case _ => off
    }
  }

  def loadElement(region: MemoryBuffer, aoff: Long, i: Int): Long =
    loadElement(region, aoff, region.loadInt(aoff), i)

  def allocate(region: MemoryBuffer, length: Int): Long = {
    region.align(contentsAlignment)
    region.allocate(contentsByteSize(length))
  }

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val eltOrd = elementType.unsafeOrdering(missingGreatest)

    new UnsafeOrdering {
      override def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
        val length1 = loadLength(r1, o1)
        val length2 = loadLength(r2, o2)

        var i = 0
        while (i < math.min(length1, length2)) {
          val leftDefined = isElementDefined(r1, o1, i)
          val rightDefined = isElementDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val eOff1 = loadElement(r1, o1, length1, i)
            val eOff2 = loadElement(r2, o2, length2, i)
            val c = eltOrd.compare(r1, eOff1, r2, eOff2)
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            if (missingGreatest)
              return c
            else
              return -c
          }
          i += 1
        }
        Integer.compare(length1, length2)
      }
    }
  }

  override lazy val fundamentalType: TArray = TArray(elementType.fundamentalType)
}

abstract class TIterable extends TContainer {

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && (a1.asInstanceOf[Iterable[_]].size == a2.asInstanceOf[Iterable[_]].size)
      && a1.asInstanceOf[Iterable[_]].zip(a2.asInstanceOf[Iterable[_]])
      .forall { case (e1, e2) => elementType.valuesSimilar(e1, e2, tolerance) })
}

final case class TArray(elementType: Type) extends TIterable {
  override def toString = s"Array[$elementType]"

  override def canCompare(other: Type): Boolean = other match {
    case TArray(otherType) => elementType.canCompare(otherType)
    case _ => false
  }

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

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(extendOrderingToNull(missingGreatest)(
      Ordering.Iterable(elementType.ordering(missingGreatest))))

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

final case class TSet(elementType: Type) extends TIterable {
  override def toString = s"Set[$elementType]"

  override def canCompare(other: Type): Boolean = other match {
    case TSet(otherType) => elementType.canCompare(otherType)
    case _ => false
  }

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

  def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val elementSortOrd = elementType.ordering(true)
    val itOrd = Ordering.Iterable(elementType.ordering(missingGreatest))
    val setOrdering = new Ordering[Set[Annotation]] {
      def compare(x: Set[Annotation], y: Set[Annotation]): Int = {
        val s1 = x.toArray.sorted(elementSortOrd)
        val s2 = y.toArray.sorted(elementSortOrd)

        itOrd.compare(s1, s2)
      }
    }

    annotationOrdering(extendOrderingToNull(missingGreatest)(setOrdering))
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

final case class TDict(keyType: Type, valueType: Type) extends TContainer {

  override def canCompare(other: Type): Boolean = other match {
    case TDict(okt, ovt) => keyType.canCompare(okt) && valueType.canCompare(ovt)
    case _ => false
  }

  val elementType: Type = TStruct("key" -> keyType, "value" -> valueType)

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
    a1 == a2 || (a1 != null && a2 != null &&
      a1.asInstanceOf[Map[Any, _]].outerJoin(a2.asInstanceOf[Map[Any, _]])
        .forall { case (_, (o1, o2)) =>
          o1.liftedZip(o2).exists { case (v1, v2) => valueType.valuesSimilar(v1, v2, tolerance) }
        })

  override def desc: String =
    """
    A ``Dict`` is an unordered collection of key-value pairs. Each key can only appear once in the collection.
    """

  override def scalaClassTag: ClassTag[Map[_, _]] = classTag[Map[_, _]]

  def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val elementSortOrd = elementType.ordering(true)
    val itOrd = Ordering.Iterable(elementType.ordering(missingGreatest))
    val dict = new Ordering[Map[Annotation, Annotation]] {
      def compare(x: Map[Annotation, Annotation], y: Map[Annotation, Annotation]): Int = {
        val s1 = x.toArray.map { case (k, v) => Row(k, v) }.sorted(elementSortOrd)
        val s2 = y.toArray.map { case (k, v) => Row(k, v) }.sorted(elementSortOrd)

        itOrd.compare(s1, s2)
      }
    }

    annotationOrdering(extendOrderingToNull(missingGreatest)(dict))
  }
}

case object TGenotype extends ComplexType {
  override def toString = "Genotype"

  override lazy val representation: TStruct = TStruct(
    "gt" -> TInt32,
    "ad" -> TArray(TInt32),
    "dp" -> TInt32,
    "gq" -> TInt32,
    "px" -> TArray(TInt32),
    "fakeRef" -> TBoolean,
    "isLinearScale" -> TBoolean)

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Genotype]

  override def genNonmissingValue: Gen[Annotation] = Genotype.genArb

  override def desc: String = "A ``Genotype`` is a Hail data type representing a genotype in the Variant Dataset. It is referred to as ``g`` in the expression language."

  override def scalaClassTag: ClassTag[Genotype] = classTag[Genotype]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val rowOrd = fundamentalType.ordering(missingGreatest)
    val ord = new Ordering[Annotation] {
      def compare(x: Annotation, y: Annotation): Int = rowOrd.compare(
        Genotype.toRow(x.asInstanceOf[Genotype]),
        Genotype.toRow(y.asInstanceOf[Genotype]))
    }

    annotationOrdering(extendOrderingToNull(missingGreatest)(ord))

  }
}

case object TCall extends ComplexType {
  override def toString = "Call"

  override lazy val representation: Type = TInt32

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = Call.genArb

  override def desc: String = "A ``Call`` is a Hail data type representing a genotype call (ex: 0/0) in the Variant Dataset."

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Int]]))
}

case object TAltAllele extends ComplexType {
  override def toString = "AltAllele"

  def typeCheck(a: Any): Boolean = a == null || a == null || a.isInstanceOf[AltAllele]

  override def genNonmissingValue: Gen[Annotation] = AltAllele.gen

  override def desc: String = "An ``AltAllele`` is a Hail data type representing an alternate allele in the Variant Dataset."

  override def scalaClassTag: ClassTag[AltAllele] = classTag[AltAllele]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[AltAllele]]))

  override lazy val representation: TStruct = TStruct(
    "ref" -> TString,
    "alt" -> TString)
}

case class TVariant(gr: GenomeReference) extends ComplexType {
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
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Variant]]))

  override val partitionKey: Type = TLocus(gr)

  override def orderedKey: OrderedKey[Annotation, Annotation] = new OrderedKey[Annotation, Annotation] {
    def project(key: Annotation): Annotation = key.asInstanceOf[Variant].locus

    val kOrd: Ordering[Annotation] = ordering(missingGreatest = true)

    val pkOrd: Ordering[Annotation] = TLocus(gr).ordering(missingGreatest = true)

    val kct: ClassTag[Annotation] = classTag[Annotation]

    val pkct: ClassTag[Annotation] = classTag[Annotation]
  }

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val fundamentalComparators = representation.fields.map(_.typ.unsafeOrdering(missingGreatest)).toArray
    val repr = representation.fundamentalType
    new UnsafeOrdering {
      def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
        val cOff1 = repr.loadField(r1, o1, 0)
        val cOff2 = repr.loadField(r2, o2, 0)

        val contig1 = TString.loadString(r1, cOff1)
        val contig2 = TString.loadString(r2, cOff2)

        val c = Contig.compare(contig1, contig2)
        if (c != 0)
          return c

        var i = 1
        while (i < representation.size) {
          val fOff1 = repr.loadField(r1, o1, i)
          val fOff2 = repr.loadField(r2, o2, i)

          val c = fundamentalComparators(i).compare(r1, fOff1, r2, fOff2)
          if (c != 0)
            return c

          i += 1
        }
        0
      }
    }
  }

  override lazy val representation: TStruct = TStruct(
    "contig" -> TString,
    "start" -> TInt32,
    "ref" -> TString,
    "altAlleles" -> TArray(TAltAllele.representation))
}

case class TLocus(gr: GenomeReference) extends ComplexType {
  override def toString = "Locus"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Locus]

  override def genNonmissingValue: Gen[Annotation] = Locus.gen

  override def desc: String = "A ``Locus`` is a Hail data type representing a specific genomic location in the Variant Dataset."

  override def scalaClassTag: ClassTag[Locus] = classTag[Locus]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Locus]]))

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val repr = representation.fundamentalType

    new UnsafeOrdering {
      def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
        val cOff1 = repr.loadField(r1, o1, 0)
        val cOff2 = repr.loadField(r2, o2, 0)

        val contig1 = TString.loadString(r1, cOff1)
        val contig2 = TString.loadString(r2, cOff2)

        val c = Contig.compare(contig1, contig2)
        if (c != 0)
          return c

        val posOff1 = repr.loadField(r1, o1, 1)
        val posOff2 = repr.loadField(r2, o2, 1)
        java.lang.Integer.compare(r1.loadInt(posOff1), r2.loadInt(posOff2))
      }
    }
  }

  override lazy val representation: TStruct = TStruct(
    "contig" -> TString,
    "position" -> TInt32)
}

case class TInterval(gr: GenomeReference) extends ComplexType {
  override def toString = "Interval"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Interval[_]] && a.asInstanceOf[Interval[_]].end.isInstanceOf[Locus]

  override def genNonmissingValue: Gen[Annotation] = Interval.gen(Locus.gen)

  override def desc: String = "An ``Interval`` is a Hail data type representing a range of genomic locations in the Variant Dataset."

  override def scalaClassTag: ClassTag[Interval[Locus]] = classTag[Interval[Locus]]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Interval[Locus]]]))

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val locusOrd = TLocus(gr).unsafeOrdering(missingGreatest)
    new UnsafeOrdering {
      def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
        val sOff1 = representation.loadField(r1, o1, 0)
        val sOff2 = representation.loadField(r2, o2, 0)

        val c1 = locusOrd.compare(r1, sOff1, r2, sOff2)
        if (c1 != 0)
          return c1

        val eOff1 = representation.loadField(r1, o1, 1)
        val eOff2 = representation.loadField(r2, o2, 1)

        locusOrd.compare(r1, eOff1, r2, eOff2)
      }
    }
  }

  override lazy val representation: TStruct = TStruct(
    "start" -> TLocus(gr).representation,
    "end" -> TLocus(gr).representation)
}

final case class Field(name: String, typ: Type,
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

final case class TStruct(fields: IndexedSeq[Field]) extends Type {
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

  def field(name: String): Field = fields(fieldIdx(name))

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

  override def queryTyped(p: List[String]): (Type, Querier) = {
    if (p.isEmpty)
      (this, identity[Annotation])
    else {
      selfField(p.head) match {
        case Some(f) =>
          val (t, q) = f.typ.queryTyped(p.tail)
          val localIndex = f.index
          (t, (a: Any) =>
            if (a == null)
              null
            else
              q(a.asInstanceOf[Row].get(localIndex)))
        case None => throw new AnnotationPathException(s"struct has no field ${ p.head }")
      }
    }
  }

  override def delete(p: List[String]): (Type, Deleter) = {
    if (p.isEmpty)
      (TStruct.empty, a => null)
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
        if (a == null)
          null
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
          case None => r.append(keyF(null, toIns))
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
        null
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
        null
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
    if (size == 0) {
      Gen.const(Annotation.empty)
    } else
      Gen.size.flatMap(fuel =>
        if (size > fuel) Gen.const(null)
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

    annotationOrdering(
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
      }))
  }

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val fieldOrderings = fields.map(_.typ.unsafeOrdering(missingGreatest)).toArray

    new UnsafeOrdering {
      def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
        var i = 0
        while (i < size) {
          val leftDefined = isFieldDefined(r1, o1, i)
          val rightDefined = isFieldDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, loadField(r1, o1, i), r2, loadField(r2, o2, i))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            if (missingGreatest)
              return c
            else
              return -c
          }
          i += 1
        }
        0
      }
    }
  }

  // needs to be lazy, because the compiler uses placeholders
  lazy val (byteOffsets, _byteSize): (Array[Long], Long) = {
    val a = new Array[Long](size)

    val bp = new BytePacker()

    val nMissingBytes: Long = (size + 7) / 8
    var offset = nMissingBytes
    fields.foreach { f =>
      val fSize = f.typ.byteSize
      val fAlignment = f.typ.alignment

      bp.getSpace(fSize, fAlignment) match {
        case Some(start) =>
          a(f.index) = start
        case None =>
          val mod = offset % fAlignment
          if (mod != 0) {
            val shift = fAlignment - mod
            bp.insertSpace(shift, offset)
            offset += (fAlignment - mod)
          }
          a(f.index) = offset
          offset += fSize
      }
    }
    a -> offset
  }

  override def byteSize: Long = _byteSize

  override lazy val alignment: Long =
    if (fields.isEmpty)
      1
    else
      fields.map(_.typ.alignment).max

  override lazy val fundamentalType: TStruct = {
    val fundamentalFieldTypes = fields.map(f => f.typ.fundamentalType)
    if ((fields, fundamentalFieldTypes).zipped
      .forall { case (f, ft) => f.typ == ft })
      this
    else
      TStruct((fields, fundamentalFieldTypes).zipped.map { case (f, ft) => (f.name, ft) }: _*)
  }

  def allocate(region: MemoryBuffer): Long = {
    region.align(alignment)
    region.allocate(byteSize)
  }

  def isFieldDefined(region: MemoryBuffer, offset: Long, fieldIdx: Int): Boolean =
    !region.loadBit(offset, fieldIdx)

  def setFieldMissing(region: MemoryBuffer, offset: Long, fieldIdx: Int) {
    region.setBit(offset, fieldIdx)
  }

  def fieldOffset(offset: Long, fieldIdx: Int): Long =
    offset + byteOffsets(fieldIdx)

  def loadField(region: MemoryBuffer, offset: Long, fieldIdx: Int): Long = {
    val off = fieldOffset(offset, fieldIdx)
    fields(fieldIdx).typ.fundamentalType match {
      case _: TArray | TBinary => region.loadAddress(off)
      case _ => off
    }
  }
}
