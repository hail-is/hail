package is.hail.expr.ir

import is.hail.expr.ir.agg._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

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
final case class ReservoirSample() extends AggOp
final case class Densify() extends AggOp
final case class TakeBy(so: SortOrder = Ascending) extends AggOp
final case class Group() extends AggOp
final case class AggElements() extends AggOp
final case class AggElementsLengthCheck() extends AggOp
final case class PrevNonnull() extends AggOp
final case class ImputeType() extends AggOp
final case class NDArraySum() extends AggOp
final case class NDArrayMultiplyAdd() extends AggOp
final case class Fold() extends AggOp

// exists === map(p).sum, needs short-circuiting aggs
// forall === map(p).product, needs short-circuiting aggs

object AggOp {
  // only to be used with virtual non-nested signatures on ApplyAggOp and ApplyScanOp
  def getReturnType(op: AggOp, seqOpArgs: IndexedSeq[Type]): Type = (op, seqOpArgs) match {
    case (Sum(), Seq(t)) => t
    case (Product(), Seq(t)) => t
    case (Min(), Seq(t)) => t
    case (Max(), Seq(t)) => t
    case (Count(), _) => TInt64
    case (Take(), Seq(t)) => TArray(t)
    case (ReservoirSample(), Seq(t)) => TArray(t)
    case (CallStats(), _) => CallStatsState.resultPType.virtualType
    case (TakeBy(_), Seq(value, _)) => TArray(value)
    case (PrevNonnull(), Seq(t)) => t
    case (CollectAsSet(), Seq(t)) => TSet(t)
    case (Collect(), Seq(t)) => TArray(t)
    case (Densify(), Seq(t)) => t
    case (ImputeType(), _) => ImputeTypeState.resultEmitType.virtualType
    case (LinearRegression(), _) =>
      LinearRegressionAggregator.resultPType.virtualType
    case (ApproxCDF(), _) => QuantilesAggregator.resultPType.virtualType
    case (Downsample(), Seq(_, _, _)) => DownsampleAggregator.resultType
    case (NDArraySum(), Seq(t)) => t
    case (NDArrayMultiplyAdd(), Seq(a: TNDArray, _)) => a
    case _ => throw new UnsupportedExtraction(this.toString)
  }

  def prune(agg: AggOp, seqOpArgs: IndexedSeq[Type], requestedType: Type): IndexedSeq[Type] =
    (agg, seqOpArgs) match {
      case (Collect(), Seq(_)) =>
        FastSeq(requestedType.asInstanceOf[TArray].elementType)
      case (Take(), Seq(_)) =>
        FastSeq(requestedType.asInstanceOf[TArray].elementType)
      case (ReservoirSample(), Seq(_)) =>
        FastSeq(requestedType.asInstanceOf[TArray].elementType)
      case (TakeBy(_), Seq(_, k)) =>
        FastSeq(requestedType.asInstanceOf[TArray].elementType, k)
      case (PrevNonnull(), Seq(_)) =>
        FastSeq(requestedType)
      case (Densify(), Seq(_)) =>
        FastSeq(requestedType)
      case _ => seqOpArgs
    }

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
    case "reservoirSample" | "ReservoirSample" => ReservoirSample()
    case "densify" | "Densify" => Densify()
    case "takeBy" | "TakeBy" => TakeBy()
    case "callStats" | "CallStats" => CallStats()
    case "linreg" | "LinearRegression" => LinearRegression()
    case "downsample" | "Downsample" => Downsample()
    case "prevnonnull" | "PrevNonnull" => PrevNonnull()
    case "Group" => Group()
    case "AggElements" => AggElements()
    case "AggElementsLengthCheck" => AggElementsLengthCheck()
    case "ImputeType" => ImputeType()
    case "NDArraySum" => NDArraySum()
    case "NDArrayMutiplyAdd" => NDArrayMultiplyAdd()
    case "Fold" => Fold()
  }
}
