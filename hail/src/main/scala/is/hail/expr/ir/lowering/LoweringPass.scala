package is.hail.expr.ir.lowering

import is.hail.expr.ir.agg.Extract
import is.hail.expr.ir._
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

case class LowerToDistributedArrayPass(t: DArrayLowering.Type) extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "LowerToDistributedArray"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LowerIR.lower(ir.asInstanceOf[IR], t)
}

case object InlineApplyIR extends LoweringPass {
  val before: IRState = CompilableIR
  val after: IRState = CompilableIRNoApply
  val context: String = "InlineApplyIR"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = RewriteBottomUp(ir, {
    case x: ApplyIR => Some(x.explicitNode)
    case _ => None
  })
}

case object LowerArrayAggsToRunAggsPass extends LoweringPass {
  val before: IRState = CompilableIRNoApply
  val after: IRState = EmittableIR
  val context: String = "LowerArrayAggsToRunAggs"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = RewriteBottomUp(ir, {
    case x@StreamAgg(a, name, query) =>
      val res = genUID()
      val aggs = Extract(query, res)
      val newNode = Let(
        res,
        RunAgg(
          Begin(FastSeq(
            aggs.init,
            StreamFor(
              a,
              name,
              aggs.seqPerElt))),
          aggs.results,
          aggs.aggs),
        aggs.postAggIR)
      if (newNode.typ != x.typ)
        throw new RuntimeException(s"types differ:\n  new: ${ newNode.typ }\n  old: ${ x.typ }")
      Some(newNode)
    case x@StreamAggScan(a, name, query) =>
      val res = genUID()
      val aggs = Extract(Extract.liftScan(query), res)
      val newNode = RunAggScan(
        a,
        name,
        aggs.init,
        aggs.seqPerElt,
        Let(res, aggs.results, aggs.postAggIR),
        aggs.aggs
      )
      if (newNode.typ != x.typ)
        throw new RuntimeException(s"types differ:\n  new: ${ newNode.typ }\n  old: ${ x.typ }")
      Some(newNode)
    case _ => None
  })
}
