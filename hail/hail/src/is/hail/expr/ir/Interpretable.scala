package is.hail.expr.ir

import is.hail.expr.ir.defs._
import is.hail.types.virtual.TNDArray

object Interpretable {
  def apply(ir: IR): Boolean =
    ir match {
      case _ if ir.typ.isInstanceOf[TNDArray] => false
      case _: RunAgg |
          _: InitOp |
          _: SeqOp |
          _: CombOp |
          _: ResultOp |
          _: CombOpValue |
          _: InitFromSerializedValue |
          _: AggStateValue |
          _: RunAggScan |
          _: SerializeAggs |
          _: DeserializeAggs |
          _: ArrayZeros |
          _: NDArrayQR |
          _: NDArraySVD |
          _: NDArrayEigh |
          _: NDArrayShape |
          _: NDArrayRef |
          _: ReadPartition |
          _: WritePartition |
          _: WriteMetadata |
          _: ReadValue |
          _: WriteValue |
          _: NDArrayWrite |
          _: StreamZipJoinProducers |
          _: ArrayMaximalIndependentSet |
          _: RNGSplit |
          _: RNGSplitStatic |
          _: RNGStateLiteral => false
      case Block(bindings, _) =>
        bindings.forall(_.scope == Scope.EVAL)
      case x: ApplyIR =>
        !Exists(
          x.body,
          {
            case n: IR => !Interpretable(n)
            case _ => false
          },
        )
      case _ => true
    }
}
