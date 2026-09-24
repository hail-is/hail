package is.hail.expr.ir.analyses

import is.hail.expr.ir.{BaseIR, IRTraversal, Memo, UsesAndDefs}
import is.hail.expr.ir.defs.{Recur, Ref, TailLoop}
import is.hail.types.virtual.TStream

import scala.annotation.tailrec

object ControlFlowPreventsSplit {
  def apply(x: BaseIR, usesAndDefs: UsesAndDefs): Memo[Unit] = {
    val m = Memo.empty[Unit]

    @tailrec def unwind(p: BaseIR => Boolean)(stack: List[BaseIR]): Unit =
      stack match {
        case x :: xs if p(x) =>
          if (!m.contains(x)) m.bind(x, ())
          unwind(p)(xs)
        case Nil =>
          throw new AssertionError("unwound past ir root")
        case _ =>
      }

    IRTraversal.trace(x).foreach { path =>
      path.head match {
        case r @ Recur(name, _, _) =>
          m.bind(r, ())
          unwind {
            case TailLoop(`name`, _, _, _) => false
            case _ => true
          }(path.tail)
        case r @ Ref(_, t) if t.isInstanceOf[TStream] =>
          m.bind(r, ())
          val declaration = usesAndDefs.defs.lookup(r)
          unwind(!_.eq(declaration))(path.tail)
        case _ =>
      }
    }

    m
  }
}
