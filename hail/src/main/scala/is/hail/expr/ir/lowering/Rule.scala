package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BaseIR, BlockMatrixMultiWrite, BlockMatrixToTableApply, BlockMatrixToValueApply, BlockMatrixWrite, Compilable, IR, InterpretableButNotCompilable, MatrixAggregate, MatrixIR, MatrixToValueApply, MatrixWrite, RelationalLet, RelationalLetBlockMatrix, RelationalLetMatrixTable, RelationalLetTable, RelationalRef, TableAggregate, TableCollect, TableCount, TableGetGlobals, TableToValueApply, TableWrite}

trait Rule {
  def allows(ir: BaseIR): Boolean
}

object NoMatrixIR extends Rule {
  def allows(ir: BaseIR): Boolean = !ir.isInstanceOf[MatrixIR]
}

object NoRelationalLets extends Rule {
  def allows(ir: BaseIR): Boolean = ir match {
    case _: RelationalLet => false
    case _: RelationalLetBlockMatrix => false
    case _: RelationalLetMatrixTable => false
    case _: RelationalLetTable => false
    case _: RelationalRef => false
    case _ => true
  }
}

object CompilableValueIRs extends Rule {
  def allows(ir: BaseIR): Boolean = ir match {
    case x: IR => Compilable(x)
    case _ => true
  }
}

object ValueIROnly extends Rule {
  def allows(ir: BaseIR): Boolean = ir match {
    case _: IR => true
    case _ => false
  }
}
