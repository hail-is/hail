package is.hail.expr.ir

import is.hail.expr.types.virtual.TVoid
import is.hail.utils._

import scala.collection.mutable

object InterpretNonCompilable {

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    RewriteBottomUp(ir, {
      case x: IR if InterpretableButNotCompilable(x) =>
        val preTime = System.nanoTime()
        log.info(s"interpreting non compilable node: $x")
        val r = Interpret.alreadyLowered(ctx, x)
        log.info(s"took ${ formatTime(System.nanoTime() - preTime) }")
        Some(if (x.typ == TVoid) {
          Begin(FastIndexedSeq())
        } else Literal.coerce(x.typ, r))
      case _ => None
    })
  }
}
