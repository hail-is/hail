package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs.{RelationalLet, RelationalRef}

import scala.collection.mutable

object ForwardRelationalLets {
  def apply(ctx: ExecuteContext, ir0: BaseIR): BaseIR =
    ctx.time {
      val uses = mutable.HashMap.empty[Name, (Int, Int)]
      val nestingDepth = NestingDepth(ctx, ir0)
      IRTraversal.levelOrder(ir0).foreach {
        case x @ RelationalRef(name, _) =>
          val (n, nd) = uses.getOrElseUpdate(name, (0, 0))
          uses(name) = (n + 1, math.max(nd, nestingDepth.lookupRef(x)))
        case _ =>
      }

      def shouldForward(name: Name): Boolean =
        uses.get(name).forall(t => t._1 < 2 && t._2 < 1)

      // short circuit if possible
      if (!uses.keys.exists(shouldForward)) ir0
      else {
        val env = mutable.HashMap.empty[Name, IR]

        def rewrite(ir1: BaseIR): BaseIR = ir1 match {
          case RelationalLet(name, value, body) =>
            if (shouldForward(name)) {
              env(name) = rewrite(value).asInstanceOf[IR]
              rewrite(body)
            } else
              RelationalLet(name, rewrite(value).asInstanceOf[IR], rewrite(body).asInstanceOf[IR])
          case RelationalLetTable(name, value, body) =>
            if (shouldForward(name)) {
              env(name) = rewrite(value).asInstanceOf[IR]
              rewrite(body)
            } else RelationalLetTable(
              name,
              rewrite(value).asInstanceOf[IR],
              rewrite(body).asInstanceOf[TableIR],
            )
          case RelationalLetMatrixTable(name, value, body) =>
            if (shouldForward(name)) {
              env(name) = rewrite(value).asInstanceOf[IR]
              rewrite(body)
            } else RelationalLetMatrixTable(
              name,
              rewrite(value).asInstanceOf[IR],
              rewrite(body).asInstanceOf[MatrixIR],
            )
          case x @ RelationalRef(name, _) =>
            env.getOrElse(name, x)
          case _ => ir1.mapChildren(rewrite)
        }

        rewrite(ir0)
      }
    }
}
