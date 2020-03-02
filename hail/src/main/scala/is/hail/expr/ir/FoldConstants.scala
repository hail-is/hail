package is.hail.expr.ir

import is.hail.expr.types.virtual.TStream
import is.hail.utils.HailException

object FoldConstants {
  def apply(ir: BaseIR): BaseIR =
    ExecuteContext.scoped { ctx =>
      RewriteBottomUp(ir, {
        case _: Ref |
             _: In |
             _: RelationalRef |
             _: RelationalLet |
             _: ApplySeeded |
             _: ApplyAggOp |
             _: ApplyScanOp |
             _: Begin |
             _: MakeNDArray |
             _: NDArrayShape |
             _: NDArrayReshape |
             _: NDArrayConcat |
             _: NDArraySlice |
             _: NDArrayFilter |
             _: NDArrayMap |
             _: NDArrayMap2 |
             _: NDArrayReindex |
             _: NDArrayAgg |
             _: NDArrayWrite |
             _: NDArrayMatMul |
             _: Die => None
        case ir: IR if ir.typ.isInstanceOf[TStream] => None
        case ir: IR if !IsConstant(ir) &&
          Interpretable(ir) &&
          ir.children.forall {
            case c: IR => IsConstant(c)
            case _ => false
          } =>
          try {
            Some(Literal.coerce(ir.typ, Interpret.alreadyLowered(ctx, ir)))
          } catch {
            case _: HailException => None
          }
        case _ => None
      })
    }
}