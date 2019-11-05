package is.hail.expr.ir

import is.hail.expr.types.virtual.TVoid
import is.hail.utils._

import scala.collection.mutable

object InterpretNonCompilable {

  def
  apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    val included = mutable.Set.empty[IR]

    def visit(ir: BaseIR): Unit = {
      ir match {
        case x: IR if !Compilable(x) && !included.contains(x) =>
          included += x
        case _ =>
      }
      ir.children.foreach {
        case ir: IR => visit(ir)
        case _ =>
      }
    }

    visit(ir)

    if (included.isEmpty)
      return ir

    val m = mutable.HashMap.empty[IR, IR]
    included.foreach { toEvaluate =>
      val preTime = System.nanoTime()
      log.info(s"interpreting non compilable node: $toEvaluate")
      val r = Interpret.alreadyLowered(ctx, toEvaluate)
      log.info(s"took ${ formatTime(System.nanoTime() - preTime) }")
      val lit = if (toEvaluate.typ == TVoid) {
        Begin(FastIndexedSeq())
      } else Literal.coerce(toEvaluate.typ, r)
      m(toEvaluate) = lit
    }

    def rewrite(x: BaseIR): BaseIR = {
      val replacement = x match {
        case ir: IR => m.get(ir)
        case _ => None
      }
      replacement match {
        case Some(r) => r
        case None =>
          val children = x.children
          val rewritten = children.map(rewrite)
          val refEq = children.indices.forall(i => children(i).eq(rewritten(i)))
          if (refEq)
            x
          else
            x.copy(rewritten)
      }
    }

    rewrite(ir)
  }
}
