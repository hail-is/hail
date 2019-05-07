package is.hail.expr.ir

import is.hail.utils.HailException

object FoldConstants {
  def apply(ir: BaseIR, canGenerateLiterals: Boolean = true): BaseIR =
    RewriteBottomUp(ir, {
      case ir: IR if !IsConstant(ir) &&
        Interpretable(ir) &&
        ir.children.forall {
          case c: IR => IsConstant(c)
          case _ => false
        } &&
        (canGenerateLiterals || CanEmit(ir.typ)) =>
        try {
          Some(
            Literal.coerce(ir.typ, Interpret(ir, optimize = false)))
        } catch {
          case _: HailException => None
        }
    })
}
