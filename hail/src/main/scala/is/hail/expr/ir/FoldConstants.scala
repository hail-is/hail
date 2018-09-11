package is.hail.expr.ir

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
               _: InitOp => None
          case Let(name, value, body) if IsConstant(value) =>
            Some(FoldConstants(Subst(body, Env.empty[IR].bind(name, value)), canGenerateLiterals))
          case _ =>
            if (ir.children.forall {
              case c: IR => IsConstant(c)
              case _ => false
            } && (canGenerateLiterals || CanEmit(ir.typ)))
              Some(Literal.coerce(ir.typ, Interpret(ir, optimize = false)))
            else None
        }
      case _ => None
    })
}