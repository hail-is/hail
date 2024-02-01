package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex}
import is.hail.expr.ir._
import is.hail.types._
import is.hail.utils
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.reflect.ClassTag

import org.apache.spark.sql.types.DataType
import org.json4s.{CustomSerializer, JValue}
import org.json4s.JsonAST.JString

class TypeSerializer extends CustomSerializer[Type](format =>
      (
        { case JString(s) => IRParser.parseType(s) },
        { case t: Type => JString(t.parsableString()) },
      )
    )

object Type {
  def genScalar(): Gen[Type] =
    Gen.oneOf(TBoolean, TInt32, TInt64, TFloat32,
      TFloat64, TString, TCall)

  def genComplexType(): Gen[Type] = {
    val rgDependents = ReferenceGenome.hailReferences.toArray.map(TLocus(_))
    val others = Array(TCall)
    Gen.oneOfSeq(rgDependents ++ others)
  }

  def genFields(genFieldType: Gen[Type]): Gen[Array[Field]] = {
    Gen.buildableOf[Array](
      Gen.zip(Gen.identifier, genFieldType)
    )
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields =>
        fields
          .iterator
          .zipWithIndex
          .map { case ((k, t), i) => Field(k, t, i) }
          .toArray
      )
  }

  def preGenStruct(genFieldType: Gen[Type]): Gen[TStruct] =
    for (fields <- genFields(genFieldType)) yield TStruct(fields)

  def preGenTuple(genFieldType: Gen[Type]): Gen[TTuple] =
    for (fields <- genFields(genFieldType)) yield TTuple(fields.map(_.typ): _*)

  private val defaultRequiredGenRatio = 0.2
  def genStruct: Gen[TStruct] = Gen.coin(defaultRequiredGenRatio).flatMap(c => preGenStruct(genArb))

  def genSized(size: Int, genTStruct: Gen[TStruct]): Gen[Type] =
    if (size < 1)
      Gen.const(TStruct.empty)
    else if (size < 2)
      genScalar()
    else {
      Gen.frequency(
        (4, genScalar()),
        (1, genComplexType()),
        (
          1,
          genArb.map {
            TArray(_)
          },
        ),
        (
          1,
          genArb.map {
            TSet(_)
          },
        ),
        (
          1,
          genArb.map {
            TInterval(_)
          },
        ),
        (1, preGenTuple(genArb)),
        (1, Gen.zip(genRequired, genArb).map { case (k, v) => TDict(k, v) }),
        (1, genTStruct.resize(size)),
      )
    }

  def preGenArb(genStruct: Gen[TStruct] = genStruct): Gen[Type] =
    Gen.sized(genSized(_, genStruct))

  def genArb: Gen[Type] = preGenArb()

  val genOptional: Gen[Type] = preGenArb()

  val genRequired: Gen[Type] = preGenArb()

  def genWithValue(sm: HailStateManager): Gen[(Type, Annotation)] = for {
    s <- Gen.size
    // prefer smaller type and bigger values
    fraction <- Gen.choose(0.1, 0.3)
    x = (fraction * s).toInt
    y = s - x
    t <- Type.genStruct.resize(x)
    v <- t.genValue(sm).resize(y)
  } yield (t, v)

  implicit def arbType = Arbitrary(genArb)
}

abstract class Type extends BaseType with Serializable {
  self =>

  def children: IndexedSeq[Type] = FastSeq()

  def clear(): Unit = children.foreach(_.clear())

  def unify(concrete: Type): Boolean =
    this == concrete

  def _isCanonical: Boolean = true

  final def isCanonical: Boolean = _isCanonical && children.forall(_.isCanonical)

  def isPrimitive: Boolean = this match {
    case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean => true
    case _ => false
  }

  def isBound: Boolean = children.forall(_.isBound)

  def subst(): Type = this

  def query(fields: String*): Querier = query(fields.toList)

  def query(path: List[String]): Querier = {
    val (_, q) = queryTyped(path)
    q
  }

  def queryTyped(fields: String*): (Type, Querier) = queryTyped(fields.toList)

  def queryTyped(path: List[String]): (Type, Querier) =
    if (path.nonEmpty)
      throw new AnnotationPathException(s"invalid path ${path.mkString(".")} from type ${this}")
    else
      (this, identity[Annotation])

  final def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    _pretty(sb, indent, compact)

  def _toPretty: String

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    sb.append(_toPretty)

  def schema: DataType = SparkAnnotationImpex.exportType(this)

  def str(a: Annotation): String = if (a == null) "NA" else a.toString

  def _showStr(a: Annotation): String = str(a)

  def showStr(a: Annotation): String = if (a == null) "NA" else _showStr(a)

  def showStr(a: Annotation, trunc: Int): String = {
    val s = showStr(a)
    if (s.length > trunc)
      s.substring(0, trunc - 3) + "..."
    else
      s
  }

  def toJSON(a: Annotation): JValue = JSONAnnotationImpex.exportAnnotation(a, this)

  def genNonmissingValue(sm: HailStateManager): Gen[Annotation]

  def genValue(sm: HailStateManager): Gen[Annotation] =
    Gen.nextCoin(0.05).flatMap(isEmpty => if (isEmpty) Gen.const(null) else genNonmissingValue(sm))

  def isRealizable: Boolean = children.forall(_.isRealizable)

  /* compare values for equality, but compare Float and Double values by the absolute value of their
   * difference is within tolerance or with D_== */
  def valuesSimilar(
    a1: Annotation,
    a2: Annotation,
    tolerance: Double = utils.defaultTolerance,
    absolute: Boolean = false,
  ): Boolean = a1 == a2

  def scalaClassTag: ClassTag[_ <: AnyRef]

  def canCompare(other: Type): Boolean = this == other

  def mkOrdering(sm: HailStateManager, missingEqual: Boolean = true): ExtendedOrdering

  @transient protected var ord: ExtendedOrdering = _

  def ordering(sm: HailStateManager): ExtendedOrdering = {
    if (ord == null) ord = mkOrdering(sm)
    ord
  }

  def jsonReader: JSONReader[Annotation] = new JSONReader[Annotation] {
    def fromJSON(a: JValue): Annotation = JSONAnnotationImpex.importAnnotation(a, self)
  }

  def jsonWriter: JSONWriter[Annotation] = new JSONWriter[Annotation] {
    def toJSON(pk: Annotation): JValue = JSONAnnotationImpex.exportAnnotation(pk, self)
  }

  def _typeCheck(a: Any): Boolean

  final def typeCheck(a: Any): Boolean = a == null || _typeCheck(a)

  def valueSubsetter(subtype: Type): Any => Any = {
    assert(this == subtype)
    identity
  }

  def canCastTo(t: Type): Boolean = this match {
    case TInterval(tt1) => t match {
        case TInterval(tt2) => tt1.canCastTo(tt2)
        case _ => false
      }
    case TStruct(f1) => t match {
        case TStruct(f2) =>
          f1.size == f2.size && f1.indices.forall(i => f1(i).typ.canCastTo(f2(i).typ))
        case _ => false
      }
    case TTuple(f1) => t match {
        case TTuple(f2) =>
          f1.size == f2.size && f1.indices.forall(i => f1(i).typ.canCastTo(f2(i).typ))
        case _ => false
      }
    case TArray(t1) => t match {
        case TArray(t2) => t1.canCastTo(t2)
        case _ => false
      }
    case TSet(t1) => t match {
        case TSet(t2) => t1.canCastTo(t2)
        case _ => false
      }
    case TDict(k1, v1) => t match {
        case TDict(k2, v2) => k1.canCastTo(k2) && v1.canCastTo(v2)
        case _ => false
      }
    case _ => this == t
  }
}
