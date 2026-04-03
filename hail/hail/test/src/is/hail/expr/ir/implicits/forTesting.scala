package is.hail.expr.ir.implicits

import is.hail.expr.ir.BaseIR

object forTesting {
  implicit def toRichBaseIr(ir: BaseIR): RichBaseIR =
    new RichBaseIR(ir)
}
