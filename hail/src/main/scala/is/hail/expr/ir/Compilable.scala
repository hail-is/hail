package is.hail.expr.ir

object Compilable {
  def apply(ir: IR): Boolean = {
    ir match {
      case _: TableCount => false
      case _: TableGetGlobals => false
      case _: TableCollect => false
      case _: TableAggregate => false
      case _: MatrixAggregate => false
      case _: TableWrite => false
      case _: MatrixWrite => false
      case _: BlockMatrixWrite => false
      case _: BlockMatrixMultiWrite => false
      case _: TableToValueApply => false
      case _: MatrixToValueApply => false
      case _: BlockMatrixToValueApply => false
      case _: BlockMatrixToTableApply => false
      case _: RelationalRef => false
      case _: RelationalLet => false
      case _ => true
    }
  }
}
