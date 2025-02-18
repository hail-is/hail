package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.agg.Extract
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.defs.{
  ApplyIR, Begin, Let, RunAgg, RunAggScan, StreamAgg, StreamAggScan, StreamFor,
}
import is.hail.utils._

final class IrMetadata() {
  private[this] var hashCounter: Int = 0
  private[this] var markCounter: Int = 0

  var semhash: Option[SemanticHash.Type] = None

  def nextHash: Option[SemanticHash.Type] = {
    hashCounter += 1
    semhash.map(SemanticHash.extend(_, SemanticHash.Bytes.fromInt(hashCounter)))
  }

  def nextFlag: Int = {
    markCounter += 1
    markCounter
  }
}

trait LoweringPass {
  val before: IRState
  val after: IRState
  val context: String

  final def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.timer.time(context) {
      ctx.timer.time("Verify")(before.verify(ir))
      val result = ctx.timer.time("Transform")(transform(ctx, ir))
      ctx.timer.time("Verify")(after.verify(result))
      result
    }

  protected def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR
}

case class OptimizePass(_context: String) extends LoweringPass {
  override val context = s"Optimize: ${_context}"
  override val before: IRState = AnyIR
  override val after: IRState = AnyIR
  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = Optimize(ctx, ir)
}

case object LowerMatrixToTablePass extends LoweringPass {
  val before: IRState = AnyIR
  val after: IRState = MatrixLoweredToTable
  val context: String = "LowerMatrixToTable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = ir match {
    case x: IR => LowerMatrixIR(ctx, x)
    case x: TableIR => LowerMatrixIR(ctx, x)
    case x: MatrixIR => LowerMatrixIR(ctx, x)
    case x: BlockMatrixIR => LowerMatrixIR(ctx, x)
  }
}

case object LiftRelationalValuesToRelationalLets extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = MatrixLoweredToTable
  val context: String = "LiftRelationalValuesToRelationalLets"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LiftRelationalValues(ir)
}

case object LegacyInterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = ExecutableTableIR
  val context: String = "InterpretNonCompilable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LowerOrInterpretNonCompilable(ctx, ir)
}

case object LowerOrInterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "LowerOrInterpretNonCompilable"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LowerOrInterpretNonCompilable(ctx, ir)
}

case class LowerToDistributedArrayPass(t: DArrayLowering.Type) extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "LowerToDistributedArray"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LowerToCDA(ir.asInstanceOf[IR], t, ctx)
}

case object InlineApplyIR extends LoweringPass {
  val before: IRState = CompilableIR
  val after: IRState = CompilableIRNoApply
  val context: String = "InlineApplyIR"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = RewriteBottomUp(
    ir,
    {
      case x: ApplyIR => Some(x.explicitNode)
      case _ => None
    },
  )
}

case object LowerArrayAggsToRunAggsPass extends LoweringPass {
  val before: IRState = CompilableIRNoApply
  val after: IRState = EmittableIR
  val context: String = "LowerArrayAggsToRunAggs"

  def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    val x = ir.noSharing(ctx)
    val r = Requiredness(x, ctx)
    RewriteBottomUp(
      x,
      {
        case x @ StreamAgg(a, name, query) =>
          val aggs = Extract(query, r)

          val newNode = aggs.rewriteFromInitBindingRoot { root =>
            Let(
              FastSeq(
                aggs.resultRef.name -> RunAgg(
                  Begin(FastSeq(
                    aggs.init,
                    StreamFor(a, name, aggs.seqPerElt),
                  )),
                  aggs.results,
                  aggs.states,
                )
              ),
              root,
            )
          }

          if (newNode.typ != x.typ)
            throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")
          Some(newNode.noSharing(ctx))
        case x @ StreamAggScan(a, name, query) =>
          val aggs = Extract(query, r, isScan = true)
          val newNode = aggs.rewriteFromInitBindingRoot { root =>
            RunAggScan(
              a,
              name,
              aggs.init,
              aggs.seqPerElt,
              Let(FastSeq(aggs.resultRef.name -> aggs.results), root),
              aggs.states,
            )
          }
          if (newNode.typ != x.typ)
            throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")
          Some(newNode.noSharing(ctx))
        case _ => None
      },
    )
  }
}

case class EvalRelationalLetsPass(passesBelow: LoweringPipeline) extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = before + NoRelationalLetsState
  val context: String = "EvalRelationalLets"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    EvalRelationalLets(ir, ctx, passesBelow)
}

case class LowerAndExecuteShufflesPass(passesBelow: LoweringPipeline) extends LoweringPass {
  val before: IRState = NoRelationalLetsState + MatrixLoweredToTable
  val after: IRState = before + LoweredShuffles
  val context: String = "LowerAndExecuteShuffles"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerAndExecuteShuffles(ir, ctx, passesBelow)
}
