package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs._
import is.hail.types.virtual.{TStream, TVoid}
import is.hail.utils.HailException

object FoldConstants {
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      ctx.r.pool.scopedRegion(r => ctx.local(r = r)(foldConstants(_, ir)))
    }

  private def foldConstants(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    RewriteBottomUp(
      ir,
      {
        case _: Ref |
            _: In |
            _: RelationalRef |
            _: RelationalLet |
            _: ApplySeeded |
            _: UUID4 |
            _: ApplyAggOp |
            _: ApplyScanOp |
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
            _: Trap |
            _: Die |
            _: RNGStateLiteral => None
        case ir: IR if ir.typ.isInstanceOf[TStream] || ir.typ == TVoid => None
        case ir: IR
            if !IsConstant(ir) &&
              Interpretable(ir) &&
              ir.children.forall {
                case c: IR => IsConstant(c)
                case _ => false
              } =>
          try
            Some(Literal.coerce(ir.typ, Interpret.alreadyLowered(ctx, ir)))
          catch {
            case _: HailException => None
          }
        case _ => None
      },
    )
}
