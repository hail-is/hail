package is.hail.expr.ir

object Interpretable {
  def apply(ir: IR): Boolean = {
    ir match {
      case _: MakeNDArray | _: NDArrayRef => false
      case _ => true
    }
  }
}
