package is.hail.expr.ir

import is.hail.expr.ir.agg.Extract
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual._

object AggStateSignature {
  def apply(sig: AggSignature): AggStateSignature = AggStateSignature(Map(sig.op -> sig), sig.op)
}

case class AggStateSignature(m: Map[AggOp, AggSignature], default: AggOp, nested: Option[Seq[AggStateSignature]] = None) {
  lazy val defaultSignature: AggSignature = m(default)
  lazy val resultType: Type = Extract.getResultType(this)
  def lookup(op: AggOp): AggSignature = m(op)
}


case class AggSignature(
  op: AggOp,
  initOpArgs: Seq[Type],
  seqOpArgs: Seq[Type]) {

  def toPhysical(initOpTypes: Seq[PType], seqOpTypes: Seq[PType]): PhysicalAggSignature = {
    (initOpTypes, initOpArgs).zipped.foreach { case (pt, t) => assert(pt.virtualType == t) }
    (seqOpTypes, seqOpArgs).zipped.foreach { case (pt, t) => assert(pt.virtualType == t) }
    PhysicalAggSignature(op, initOpTypes, seqOpTypes)
  }

  lazy val singletonContainer: AggStateSignature = AggStateSignature(Map(op -> this), op, None)

  // only to be used with virtual non-nested signatures on ApplyAggOp and ApplyScanOp
  lazy val returnType: Type = AggStateSignature(this).resultType
}

case class AggStatePhysicalSignature(m: Map[AggOp, PhysicalAggSignature], default: AggOp, nested: Option[Seq[AggStatePhysicalSignature]] = None) {
  lazy val resultType: PType = Extract.getPType(this)
  lazy val defaultSignature: PhysicalAggSignature = m(default)

  lazy val virtual: AggStateSignature = AggStateSignature(m.map { case (op, p) => (op, p.virtual) }, default, nested.map(_.map(_.virtual)))

  def lookup(op: AggOp): PhysicalAggSignature = m(op)
}

object AggStatePhysicalSignature {
  def apply(sig: PhysicalAggSignature): AggStatePhysicalSignature = AggStatePhysicalSignature(Map(sig.op -> sig), sig.op)
}

case class PhysicalAggSignature(
  op: AggOp,
  physicalInitOpArgs: Seq[PType],
  physicalSeqOpArgs: Seq[PType]) {
  def initOpArgs: Seq[Type] = physicalInitOpArgs.map(_.virtualType)

  def seqOpArgs: Seq[Type] = physicalSeqOpArgs.map(_.virtualType)

  lazy val virtual: AggSignature = AggSignature(op, physicalInitOpArgs.map(_.virtualType), physicalSeqOpArgs.map(_.virtualType))
  lazy val singletonContainer: AggStatePhysicalSignature = AggStatePhysicalSignature(Map(op -> this), op, None)

}

sealed trait AggOp {}
final case class ApproxCDF() extends AggOp
final case class CallStats() extends AggOp
final case class Collect() extends AggOp
final case class CollectAsSet() extends AggOp
final case class Count() extends AggOp
final case class Downsample() extends AggOp
final case class LinearRegression() extends AggOp
final case class Max() extends AggOp
final case class Min() extends AggOp
final case class Product() extends AggOp
final case class Sum() extends AggOp
final case class Take() extends AggOp
final case class TakeBy() extends AggOp
final case class Group() extends AggOp
final case class AggElements() extends AggOp
final case class AggElementsLengthCheck() extends AggOp
final case class PrevNonnull() extends AggOp

// exists === map(p).sum, needs short-circuiting aggs
// forall === map(p).product, needs short-circuiting aggs

object AggOp {
  val fromString: PartialFunction[String, AggOp] = {
    case "approxCDF" | "ApproxCDF" => ApproxCDF()
    case "collect" | "Collect" => Collect()
    case "collectAsSet" | "CollectAsSet" => CollectAsSet()
    case "sum" | "Sum" => Sum()
    case "product" | "Product" => Product()
    case "max" | "Max" => Max()
    case "min" | "Min" => Min()
    case "count" | "Count" => Count()
    case "take" | "Take" => Take()
    case "takeBy" | "TakeBy" => TakeBy()
    case "callStats" | "CallStats" => CallStats()
    case "linreg" | "LinearRegression" => LinearRegression()
    case "downsample" | "Downsample" => Downsample()
    case "prevnonnull" | "PrevNonnull" => PrevNonnull()
    case "Group" => Group()
    case "AggElements" => AggElements()
    case "AggElementsLengthCheck" => AggElementsLengthCheck()
  }
}
