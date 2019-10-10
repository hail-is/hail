package is.hail.expr.ir

import is.hail.expr.types.virtual.TNDArray

object Interpretable {
  def apply(ir: IR): Boolean = {
    !ir.typ.isInstanceOf[TNDArray] &&
      (ir match {
      case
        _: InitOp2 |
        _: SeqOp2 |
        _: CombOp2 |
        _: ResultOp2 |
        _: SerializeAggs |
        _: DeserializeAggs |
        _: MakeNDArray |
        _: NDArrayShape |
        _: NDArrayReshape |
        _: NDArrayRef |
        _: NDArraySlice |
        _: NDArrayMap |
        _: NDArrayMap2 |
        _: NDArrayReindex |
        _: NDArrayAgg |
        _: NDArrayMatMul |
        _: NDArrayWrite => false
      case _ => true
    })
  }
}
