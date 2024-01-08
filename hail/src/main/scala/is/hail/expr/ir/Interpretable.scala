package is.hail.expr.ir

import is.hail.types.virtual.{TNDArray, TStream}

object Interpretable {
  def apply(ir: IR): Boolean = {
    !ir.typ.isInstanceOf[TNDArray] &&
    (ir match {
      case _: EncodedLiteral |
          _: RunAgg |
          _: InitOp |
          _: SeqOp |
          _: CombOp |
          _: ResultOp |
          _: CombOpValue |
          _: InitFromSerializedValue |
          _: AggStateValue |
          _: RunAgg |
          _: RunAggScan |
          _: SerializeAggs |
          _: DeserializeAggs |
          _: ArrayZeros |
          _: MakeNDArray |
          _: NDArrayShape |
          _: NDArrayReshape |
          _: NDArrayConcat |
          _: NDArrayRef |
          _: NDArraySlice |
          _: NDArrayFilter |
          _: NDArrayMap |
          _: NDArrayMap2 |
          _: NDArrayReindex |
          _: NDArrayAgg |
          _: NDArrayMatMul |
          _: TailLoop |
          _: Recur |
          _: ReadPartition |
          _: WritePartition |
          _: WriteMetadata |
          _: ReadValue |
          _: WriteValue |
          _: NDArrayWrite |
          _: StreamZipJoinProducers |
          _: ArrayMaximalIndependentSet |
          _: RNGStateLiteral => false
      case x: ApplyIR =>
        !Exists(
          x.body,
          {
            case n: IR => !Interpretable(n)
            case _ => false
          },
        )
      case _ => true
    })
  }
}
