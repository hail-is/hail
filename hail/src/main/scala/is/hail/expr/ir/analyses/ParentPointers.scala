package is.hail.expr.ir.analyses

import is.hail.expr.ir.{BaseIR, Memo}

object ParentPointers {
  def apply(x: BaseIR): Memo[BaseIR] = {
    val m = Memo.empty[BaseIR]

    def recur(ir: BaseIR, parent: BaseIR): Unit = {
      m.bind(ir, parent)
      ir.children.foreach(recur(_, ir))
    }

    recur(x, null)
    m
  }
}
