package is.hail.expr.ir.lowering

import is.hail.expr.ir.agg.Extract
import is.hail.expr.ir.{ArrayAgg, ArrayAggScan, ArrayFor, BaseIR, Begin, BlockMatrixIR, ExecuteContext, IR, InterpretNonCompilable, Let, LowerMatrixIR, MatrixIR, ResultOp, RewriteBottomUp, RunAgg, RunAggScan, TableIR, genUID}
import is.hail.utils.FastSeq

trait LoweringPass {
  val before: IRState
  val after: IRState
  val context: String

  final def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    ctx.timer.time(context) {
      ctx.timer.time("Verify")(before.verify(ir))
      val result = ctx.timer.time("LoweringTransformation")(transform(ctx: ExecuteContext, ir))
      ctx.timer.time("Verify")(after.verify(result))
      result
    }
  }

  protected def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR
}

case object LowerMatrixToTablePass extends LoweringPass {
  val before: IRState = AnyIR
  val after: IRState = MatrixLoweredToTable
  val context: String = "LowerMatrixToTable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = ir match {
    case x: IR => LowerMatrixIR(x)
    case x: TableIR => LowerMatrixIR(x)
    case x: MatrixIR => LowerMatrixIR(x)
    case x: BlockMatrixIR => LowerMatrixIR(x)
  }
}

case object LegacyInterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = ExecutableTableIR
  val context: String = "InterpretNonCompilable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = InterpretNonCompilable(ctx, ir)
}

case object InterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "InterpretNonCompilable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = InterpretNonCompilable(ctx, ir)
}

case object LowerTableToDistributedArrayPass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "LowerTableToDistributedArray"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LowerTableIR.lower(ir.asInstanceOf[IR])
}

case object LowerArrayAggsToRunAggs extends LoweringPass {
  val before: IRState = CompilableIR
  val after: IRState = EmittableIR
  val context: String = "LowerArrayAggsToRunAggs"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = RewriteBottomUp(ir, {
    case ArrayAgg(a, name, query) =>
      val res = genUID()
      val aggs = Extract(query, res)
      Some(Let(
        res,
        RunAgg(
        Begin(FastSeq(
          aggs.init,
          ArrayFor(
            a,
            name,
            aggs.seqPerElt))),
          aggs.results,
          aggs.aggs),
        aggs.postAggIR))
    case ArrayAggScan(a, name, query) =>
      val res = genUID()
      val aggs = Extract(query, res)
      Some(RunAggScan(
        a,
        name,
        aggs.init,
        aggs.seqPerElt,
        aggs.results,
        aggs.aggs
      ))
    case _ => None
  })
}