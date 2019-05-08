package is.hail.expr.ir

object Interpretable {
  def apply(ir: IR): Boolean = {
    ir match {
      case _: MakeNDArray |
           _: NDArrayReshape |
           _: NDArrayRef |
           _: NDArrayMap |
           _: NDArrayMap2 |
           _: NDArrayReindex |
           _: NDArrayAgg |
           _: NDArrayWrite => false
      case _ => true
    }
  }
}
