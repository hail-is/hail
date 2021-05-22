package is.hail.expr.ir.analyses

import is.hail.expr.ir.{BaseIR, Memo, Recur, Ref, TailLoop, UsesAndDefs, VisitIR}
import is.hail.types.virtual.TStream

object ControlFlowPreventsSplit {

  def apply(x: BaseIR, parentPointers: Memo[BaseIR], usesAndDefs: UsesAndDefs): Memo[Unit] = {
    val m = Memo.empty[Unit]
    VisitIR(x) {
      case r@Recur(name, _, _) =>
        var parent: BaseIR = r
        while (parent match {
          case TailLoop(`name`, _, _) => false
          case _ => true
        }) {
          m.bind(parent, ())
          parent = parentPointers.lookup(parent)
        }
      case r@Ref(name, t) if t.isInstanceOf[TStream] =>
        val declaration = usesAndDefs.defs.lookup(r)
        var parent: BaseIR = r
        while (!(parent.eq(declaration))) {
          m.bind(parent, ())
          parent = parentPointers.lookup(parent)
        }
      case _ =>
    }
    m
  }
}
