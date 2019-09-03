package is.hail.expr.ir

import is.hail.utils.HailException

object FoldConstants {
  def apply(ir: BaseIR, canGenerateLiterals: Boolean = true): BaseIR =
    ExecuteContext.scoped { ctx =>
      RewriteBottomUp(ir, {
        case _: Ref |
             _: In |
             _: RelationalRef |
             _: RelationalLet |
             _: ApplySeeded |
             _: ApplyAggOp |
             _: ApplyScanOp |
             _: SeqOp |
             _: Begin |
             _: InitOp |
             _: ArrayRange |
             _: MakeNDArray |
             _: NDArrayShape |
             _: NDArrayReshape |
             _: NDArraySlice |
             _: NDArraySlice |
             _: NDArrayMap |
             _: NDArrayMap2 |
             _: NDArrayReindex |
             _: NDArrayAgg |
             _: NDArrayWrite |
             _: NDArrayMatMul |
             _: Die => None
        case ir: IR if !IsConstant(ir) &&
          Interpretable(ir) &&
          ir.children.forall {
            case c: IR => IsConstant(c)
            case _ => false
          } &&
          (canGenerateLiterals || CanEmit(ir.typ)) =>
          try {
            Some(
              Literal.coerce(ir.typ, Interpret(ctx, ir, optimize = false)))
          } catch {
            case _: HailException => None
          }
        case _ => None
      })
    }
}