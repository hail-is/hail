package is.hail.expr.ir.lowering

import is.hail.expr.ir.{
  BaseIR, BlockMatrixIR, Compilable, Emittable, IR, MatrixIR, RelationalLetBlockMatrix,
  RelationalLetMatrixTable, RelationalLetTable, TableIR,
}
import is.hail.expr.ir.defs.{ApplyIR, RelationalLet, RelationalRef}

object Rules {
  type Rule = BaseIR => Boolean

  val NoMatrixIR: Rule = !_.isInstanceOf[MatrixIR]
  val NoTableIR: Rule = !_.isInstanceOf[TableIR]
  val NoBlockMatrixIR: Rule = !_.isInstanceOf[BlockMatrixIR]

  val NoRelationalLets: Rule = {
    case _: RelationalLet => false
    case _: RelationalLetBlockMatrix => false
    case _: RelationalLetMatrixTable => false
    case _: RelationalLetTable => false
    case _: RelationalRef => false
    case _ => true
  }

  val CompilableValueIRs: Rule = {
    case x: IR => Compilable(x)
    case _ => true
  }

  val NoApplyIR: Rule = !_.isInstanceOf[ApplyIR]
  val ValueIROnly: Rule = _.isInstanceOf[IR]

  val EmittableValueIRs: Rule = {
    case x: IR => Emittable(x)
    case _ => true
  }
}
