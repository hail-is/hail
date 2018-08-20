package is.hail.expr.ir

object Compilable {
  def apply(ir: IR): Boolean = {
    ir match {
      case _: TableCount => false
      case _: TableAggregate => false
      case _: TableWrite => false
      case _: TableExport  => false
      case _: MatrixWrite => false

      case _ => true
    }
  }
}
