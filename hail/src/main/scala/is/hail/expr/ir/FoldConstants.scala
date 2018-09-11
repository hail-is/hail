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
          case Let(name, value, body) if IsScalarConstant(value) =>
            Some(FoldConstants(Subst(body, Env.empty[IR].bind(name, value)), canGenerateLiterals))
          case _ =>
            if (ir.children.forall {
              case c: IR => IsScalarConstant(c)
              case _ => false
            } && (canGenerateLiterals || IsScalarType(ir.typ)))
              Some(Literal.coerce(ir.typ, Interpret(ir, optimize = false)))
            else None
        }
      case _ => None
    })
}