package is.hail.expr.ir

object Compilable {
  def selfCompilable(ir: IR): Boolean = {
    ir match {
      case _: TableCount => false
      case _: TableGetGlobals => false
      case _: TableCollect => false
      case _: TableAggregate => false
      case _: MatrixAggregate => false
      case _: TableWrite => false
      case _: TableExport  => false
      case _: MatrixWrite => false
      case _: TableToValueApply => false
      case _: MatrixToValueApply => false
      case _: Literal => false

      case _ => true
    }
  }
  def apply(ir: IR): Boolean = !Exists(ir, {
    case n: IR => !selfCompilable(n)
    case _ => true
  })
}
