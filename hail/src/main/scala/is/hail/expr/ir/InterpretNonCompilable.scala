package is.hail.expr.ir

import is.hail.expr.types.virtual.TVoid
import is.hail.utils._

import scala.collection.mutable

object InterpretNonCompilable {

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {

    def rewrite(ir: BaseIR): BaseIR = {
      val children = ir.children
      val rewritten = children.map(rewrite)
      val refEq = children.indices.forall(i => children(i).eq(rewritten(i)))
      val rwIR = if (refEq)
        ir
      else
        ir.copy(rewritten)

      rwIR match {
        case x: IR if InterpretableButNotCompilable(x) =>
          val preTime = System.nanoTime()
          log.info(s"interpreting non compilable node: $x")
          val r = Interpret.alreadyLowered(ctx, x)
          log.info(s"took ${ formatTime(System.nanoTime() - preTime) }")
          if (x.typ == TVoid) {
            Begin(FastIndexedSeq())
          } else Literal.coerce(x.typ, r)
        case _ => rwIR
      }
    }

    rewrite(ir)
  }
}
