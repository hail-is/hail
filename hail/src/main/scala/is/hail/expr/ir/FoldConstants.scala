package is.hail.expr.ir

import is.hail.utils.HailException

object FoldConstants {
  def apply(ir: BaseIR, canGenerateLiterals: Boolean = true): BaseIR =
    RewriteBottomUp(ir, {
      case ir: IR =>
        ir match {
          case _: Ref |
               _: In |
               _: ApplySeeded |
               _: ApplyAggOp |
               _: ApplyScanOp |
               _: SeqOp |
               _: Begin |
               _: InitOp |
               _: ArrayRange |
               _: Die => None
          case MakeStruct(fields) if fields.isEmpty => None
          case Let(name, value, body) if IsConstant(value) =>
            Some(FoldConstants(Subst(body, Env.empty[IR].bind(name, value)), canGenerateLiterals))
          case _ if IsConstant(ir) => None
          case _ =>
            if (ir.children.forall {
              case c: IR => IsConstant(c)
              case _ => false
            } && (canGenerateLiterals || CanEmit(ir.typ))) {
              Some(try {
                Literal.coerce(ir.typ, Interpret(ir, optimize = false))
              } catch {
                case e: HailException => Die(e.getMessage + e.getStackTrace.map(s => s"\n    $s").mkString(""), ir.typ)
              })
            }
            else None
        }
      case _ => None
    })
}