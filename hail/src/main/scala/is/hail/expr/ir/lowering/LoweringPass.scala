package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BaseIR, ExecuteContext, IR, InterpretNonCompilable, LowerMatrixIR, MatrixIR, TableIR}

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
