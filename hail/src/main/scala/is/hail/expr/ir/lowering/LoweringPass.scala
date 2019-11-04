package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BaseIR, ExecuteContext, IR, InterpretNonCompilable, LowerMatrixIR, MatrixIR, TableIR}

trait LoweringPass {
  val before: IRState
  val after: IRState
  val context: String

  final def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    before.verify(ir)
    val r = transform(ctx: ExecuteContext, ir)
    after.verify(r)
    r
  }

  protected def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR
}

case object LowerMatrixToTablePass extends LoweringPass {
  val before: IRState = AnyIR
  val after: IRState = MatrixLoweredToTable
  val context: String = "lower matrix to table"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = ir match {
    case x: IR => LowerMatrixIR(x)
    case x: TableIR => LowerMatrixIR(x)
    case x: MatrixIR => LowerMatrixIR(x)
  }
}

case object LegacyInterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = ExecutableTableIR
  val context: String = "interpret non compilable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = InterpretNonCompilable(ctx, ir)
}

case object InterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "interpret non compilable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = InterpretNonCompilable(ctx, ir)
}
