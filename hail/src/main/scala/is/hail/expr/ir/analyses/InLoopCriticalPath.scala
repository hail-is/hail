package is.hail.expr.ir.analyses

import is.hail.expr.ir.{BaseIR, Memo, Recur, TailLoop, VisitIR}

object InLoopCriticalPath {

  def apply(x: BaseIR, parentPointers: Memo[BaseIR]): Memo[Unit] = {
    val m = Memo.empty[Unit]
    VisitIR(x) {
      case r@Recur(name, _, _) =>
        var parent = parentPointers.lookup(r)
        while (parent match {
          case TailLoop(`name`, _, _) => false
          case _ => true
        }) {
          m.bind(parent, ())
          parent = parentPointers.lookup(parent)
        }
      case _ =>
    }
    m
  }
}
