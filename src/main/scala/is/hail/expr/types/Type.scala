package is.hail.expr.types

import is.hail.annotations._
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.{JSONAnnotationImpex, Parser, SparkAnnotationImpex}
import is.hail.sparkextras.OrderedKey
import is.hail.utils
import is.hail.utils._
import is.hail.variant.{GenomeReference, Variant}
import org.apache.spark.sql.types.DataType
import org.json4s.JValue

import scala.reflect.ClassTag
import scala.reflect.classTag

object Type {
  def genScalar(required: Boolean): Gen[Type] =
    Gen.oneOf(TBoolean(required), TInt32(required), TInt64(required), TFloat32(required),
      TFloat64(required), TString(required), TAltAllele(required), TCall(required))

  val genOptionalScalar: Gen[Type] = genScalar(false)

  val genRequiredScalar: Gen[Type] = genScalar(true)

  def genComplexType(required: Boolean) = {
    val grDependents = GenomeReference.references.values.toArray.flatMap(gr =>
      Array(TVariant(gr, required), TLocus(gr, required)))
    val others = Array(
      TAltAllele(required), TCall(required))
    Gen.oneOfSeq(grDependents ++ others)
  }

  val optionalComplex: Gen[Type] = genComplexType(false)

  val requiredComplex: Gen[Type] = genComplexType(true)

  def preGenStruct(required: Boolean, genFieldType: Gen[Type]): Gen[TStruct] =
    Gen.buildableOf[Array](
      Gen.zip(Gen.identifier, genFieldType))
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields => TStruct(fields
        .iterator
        .zipWithIndex
        .map { case ((k, t), i) => Field(k, t, i) }
        .toIndexedSeq))
      .map(t => if (required) (!t).asInstanceOf[TStruct] else t)

  private val defaultRequiredGenRatio = 0.2

  def genStruct: Gen[TStruct] = Gen.coin(defaultRequiredGenRatio).flatMap(preGenStruct(_, genArb))

  val genOptionalStruct: Gen[Type] = preGenStruct(required = false, genArb)

  val genRequiredStruct: Gen[Type] = preGenStruct(required = true, genArb)

  val genInsertableStruct: Gen[TStruct] = Gen.coin(defaultRequiredGenRatio).flatMap(required =>
    if (required)
      preGenStruct(required = true, genArb)
    else
      preGenStruct(required = false, genOptional))

  def genSized(size: Int, required: Boolean, genTStruct: Gen[TStruct]): Gen[Type] =
    if (size < 1)
      Gen.const(TStruct.empty(required))
    else if (size < 2)
      genScalar(required)
    else {
      Gen.frequency(
        (4, genScalar(required)),
        (1, genComplexType(required)),
        (1, genArb.map {
          TArray(_)
        }),
        (1, genArb.map {
          TSet(_)
        }),
        (1, genArb.map {
          TInterval(_)
        }),
        (1, Gen.zip(genRequired, genArb).map { case (k, v) => TDict(k, v) }),
        (1, genTStruct.resize(size)))
    }

  def preGenArb(required: Boolean, genStruct: Gen[TStruct] = genStruct): Gen[Type] =
    Gen.sized(genSized(_, required, genStruct))

  def genArb: Gen[Type] = Gen.coin(0.2).flatMap(preGenArb(_))

  val genOptional: Gen[Type] = preGenArb(required = false)

  val genRequired: Gen[Type] = preGenArb(required = true)

  val genInsertable: Gen[Type] = Gen.coin(0.2).flatMap(preGenArb(_, genInsertableStruct))

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

  def partitionKeyProjection(vType: Type): (Type, (Annotation) => Annotation) = {
    vType match {
      case t: TVariant =>
        (t.gr.locusType, (v: Annotation) => v.asInstanceOf[Variant].locus)
      case _ =>
        (vType, (a: Annotation) => a)
    }
  }
}

abstract class Type extends BaseType with Serializable {
  self =>

  def children: Seq[Type] = Seq()

  def clear(): Unit = children.foreach(_.clear())

  def desc: String = ""

  def unify(concrete: Type): Boolean = {
    this.isOfType(concrete)
  }

  def isBound: Boolean = children.forall(_.isBound)

  def subst(): Type = this.setRequired(false)

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
      (TStruct.empty(), a => null)
  }

  def unsafeInsert(typeToInsert: Type, path: List[String]): (Type, UnsafeInserter) =
    TStruct.empty().unsafeInsert(typeToInsert, path)

  def insert(signature: Type, fields: String*): (Type, Inserter) = insert(signature, fields.toList)

  def insert(signature: Type, path: List[String]): (Type, Inserter) = {
    if (path.nonEmpty)
      TStruct.empty().insert(signature, path)
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

  def _toString: String

  final override def toString = {
    if (required) "!" else ""
  } + _toString

  def _pretty(sb: StringBuilder, indent: Int = 0, compact: Boolean = false) {
    sb.append(_toString)
  }

  final def pretty(sb: StringBuilder, indent: Int = 0, compact: Boolean = false) {
    if (required)
      sb.append("!")
    _pretty(sb, indent, compact)
  }

  def toPrettyString(indent: Int = 0, compact: Boolean = false): String = {
    val sb = new StringBuilder
    pretty(sb, indent, compact = compact)
    sb.result()
  }

  def fieldOption(fields: String*): Option[Field] = fieldOption(fields.toList)

  def fieldOption(path: List[String]): Option[Field] =
    None

  def schema: DataType = SparkAnnotationImpex.exportType(this)

  def str(a: Annotation): String = if (a == null) "NA" else a.toString

  def toJSON(a: Annotation): JValue = JSONAnnotationImpex.exportAnnotation(a, this)

  def genNonmissingValue: Gen[Annotation] = ???

  def genValue: Gen[Annotation] =
    if (required)
      genNonmissingValue
    else
      Gen.nextCoin(0.05).flatMap(isEmpty => if (isEmpty) Gen.const(null) else genNonmissingValue)

  def isRealizable: Boolean = children.forall(_.isRealizable)

  /* compare values for equality, but compare Float and Double values using D_== */
  def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double = utils.defaultTolerance): Boolean = a1 == a2

  def scalaClassTag: ClassTag[_ <: AnyRef]

  def canCompare(other: Type): Boolean = this == other

  val ordering: ExtendedOrdering

  val partitionKey: Type = this

  def typedOrderedKey[PK, K] = new OrderedKey[PK, K] {
    def project(key: K): PK = key.asInstanceOf[PK]

    val kOrd: Ordering[K] = ordering.toOrdering.asInstanceOf[Ordering[K]]

    val pkOrd: Ordering[PK] = ordering.toOrdering.asInstanceOf[Ordering[PK]]

    val kct: ClassTag[K] = scalaClassTag.asInstanceOf[ClassTag[K]]

    val pkct: ClassTag[PK] = scalaClassTag.asInstanceOf[ClassTag[PK]]
  }

  def orderedKey: OrderedKey[Annotation, Annotation] = new OrderedKey[Annotation, Annotation] {
    def project(key: Annotation): Annotation = key

    val kOrd: Ordering[Annotation] = ordering.toOrdering

    val pkOrd: Ordering[Annotation] = ordering.toOrdering

    val kct: ClassTag[Annotation] = classTag[Annotation]

    val pkct: ClassTag[Annotation] = classTag[Annotation]
  }

  def jsonReader: JSONReader[Annotation] = new JSONReader[Annotation] {
    def fromJSON(a: JValue): Annotation = JSONAnnotationImpex.importAnnotation(a, self)
  }

  def jsonWriter: JSONWriter[Annotation] = new JSONWriter[Annotation] {
    def toJSON(pk: Annotation): JValue = JSONAnnotationImpex.exportAnnotation(pk, self)
  }

  def byteSize: Long = 1

  def alignment: Long = byteSize

  /*  Fundamental types are types that can be handled natively by RegionValueBuilder: primitive
      types, Array and Struct. */
  def fundamentalType: Type = this

  def required: Boolean

  def _typeCheck(a: Any): Boolean

  final def typeCheck(a: Any): Boolean = (!required && a == null) || _typeCheck(a)

  final def setRequired(required: Boolean): Type = if (this.required == required) this else !this

  final def unary_!(): Type = {
    this match {
      case TBinary(req) => TBinary(!req)
      case TBoolean(req) => TBoolean(!req)
      case TInt32(req) => TInt32(!req)
      case TInt64(req) => TInt64(!req)
      case TFloat32(req) => TFloat32(!req)
      case TFloat64(req) => TFloat64(!req)
      case TString(req) => TString(!req)
      case TCall(req) => TCall(!req)
      case TAltAllele(req) => TAltAllele(!req)
      case t: TArray => t.copy(required = !t.required)
      case t: TSet => t.copy(required = !t.required)
      case t: TDict => t.copy(required = !t.required)
      case t: TVariant => t.copy(required = !t.required)
      case t: TLocus => t.copy(required = !t.required)
      case t: TInterval => t.copy(required = !t.required)
      case t: TStruct => t.copy(required = !t.required)
    }
  }

  final def isOfType(t: Type): Boolean = {
    this match {
      case TBinary(_) => t == TBinaryOptional || t == TBinaryRequired
      case TBoolean(_) => t == TBooleanOptional || t == TBooleanRequired
      case TInt32(_) => t == TInt32Optional || t == TInt32Required
      case TInt64(_) => t == TInt64Optional || t == TInt64Required
      case TFloat32(_) => t == TFloat32Optional || t == TFloat32Required
      case TFloat64(_) => t == TFloat64Optional || t == TFloat64Required
      case TString(_) => t == TStringOptional || t == TStringRequired
      case TCall(_) => t == TCallOptional || t == TCallRequired
      case TAltAllele(_) => t == TAltAlleleOptional || t == TAltAlleleRequired
      case t2: TLocus => t == t2 || t == !t2
      case t2: TVariant => t == t2 || t == !t2
      case t2: TInterval => t == t2 || t == !t2
      case t2: TStruct =>
        t.isInstanceOf[TStruct] &&
          t.asInstanceOf[TStruct].size == t2.size &&
          t.asInstanceOf[TStruct].fields.zip(t2.fields).forall { case (f1: Field, f2: Field) => f1.typ.isOfType(f2.typ) && f1.name == f2.name }
      case t2: TArray => t.isInstanceOf[TArray] && t.asInstanceOf[TArray].elementType.isOfType(t2.elementType)
      case t2: TSet => t.isInstanceOf[TSet] && t.asInstanceOf[TSet].elementType.isOfType(t2.elementType)
      case t2: TDict => t.isInstanceOf[TDict] && t.asInstanceOf[TDict].keyType.isOfType(t2.keyType) && t.asInstanceOf[TDict].valueType.isOfType(t2.valueType)
    }
  }

  def deepOptional(): Type =
    this match {
      case t: TArray => TArray(t.elementType.deepOptional())
      case t: TSet => TSet(t.elementType.deepOptional())
      case t: TDict => TDict(t.keyType.deepOptional(), t.valueType.deepOptional())
      case t: TStruct =>
        TStruct(t.fields.map(f => Field(f.name, f.typ.deepOptional(), f.index)))
      case t =>
        t.setRequired(false)
    }

  def structOptional(): Type =
    this match {
      case t: TStruct =>
        TStruct(t.fields.map(f => Field(f.name, f.typ.deepOptional(), f.index)))
      case t =>
        t.setRequired(false)
    }
}
