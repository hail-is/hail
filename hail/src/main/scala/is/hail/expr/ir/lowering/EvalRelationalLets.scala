package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, CompileAndEvaluate, IR, RelationalLet, RelationalLetMatrixTable, RelationalLetTable, RelationalRef, RewriteBottomUp}

object EvalRelationalLets {
  // need to run the rest of lowerings to eval.
  def apply(ir: BaseIR, ctx: ExecuteContext, passesBelow: LoweringPipeline): BaseIR = {
    def execute(value: BaseIR, letsAbove: Map[String, IR]): IR = {
      val compilable = passesBelow.apply(ctx, lower(value, letsAbove))
        .asInstanceOf[IR]
      CompileAndEvaluate.evalToIR(ctx, compilable, true)
    }

    def lower(ir: BaseIR, letsAbove: Map[String, IR]): BaseIR = {
      ir match {
        case RelationalLet(name, value, body) =>
          val valueLit = execute(value, letsAbove)
          lower(body, letsAbove + (name -> valueLit))
        case RelationalLetTable(name, value, body) =>
          val valueLit = execute(value, letsAbove)
          lower(body, letsAbove + (name -> valueLit))
        case RelationalLetMatrixTable(name, value, body) =>
          val valueLit = execute(value, letsAbove)
          lower(body, letsAbove + (name -> valueLit))
        case RelationalRef(name, _) => letsAbove(name)
        case x =>
          val children = x.children
          val newChildren = children.map(lower(_, letsAbove))
          if (RewriteBottomUp.areObjectEqual(children, newChildren))
            x
          else
            x.copy(newChildren)
      }
    }

    lower(ir, Map())
  }
}
