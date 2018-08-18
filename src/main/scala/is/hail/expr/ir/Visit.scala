package is.hail.expr.ir

object Visit {
  def apply(ir: BaseIR, f: BaseIR => Unit): Unit = {
    f(ir)
    ir.children.foreach(apply(_, f))
  }
}
