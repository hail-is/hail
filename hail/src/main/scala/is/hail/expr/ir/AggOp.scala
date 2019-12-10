package is.hail.expr.ir

import is.hail.expr.types.virtual._

case class AggSignature(
  op: AggOp,
  initOpArgs: Seq[Type],
  seqOpArgs: Seq[Type],
  nested: Option[Seq[AggSignature]]) {
  lazy val returnType: Type = agg.Extract.getType(this)
}

sealed trait AggOp { }
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
  }
}
