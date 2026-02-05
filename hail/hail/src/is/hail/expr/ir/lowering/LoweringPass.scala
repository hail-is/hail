package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.agg.Extract
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.defs.{
  ApplyIR, Begin, Let, RunAgg, RunAggScan, StreamAgg, StreamAggScan, StreamFor,
}
import is.hail.expr.ir.lowering.Invariant._
import is.hail.utils.implicits.toRichPredicate

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

abstract class LoweringPass(implicit E: sourcecode.Enclosing) {

  val before: Invariant
  val after: Invariant
  val context: String

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      before.verify(ctx, ir)
      val result = transform(ctx, ir)
      after.verify(ctx, result)
      result
    }

  protected def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR
}

case class OptimizePass(_context: String) extends LoweringPass {
  override val context = s"Optimize: ${_context}"
  override val before: Invariant = AnyIR
  override val after: Invariant = AnyIR

  override def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    if (ctx.flags.isDefined(Optimize.Flags.Optimize)) super.apply(ctx, ir)
    else ir

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = Optimize(ctx, ir)
}

case object LowerMatrixToTablePass extends LoweringPass {
  val before: Invariant = AnyIR
  val after: Invariant = NoMatrixIR
  val context: String = "LowerMatrixToTable"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = ir match {
    case x: IR => LowerMatrixIR(ctx, x)
    case x: TableIR => LowerMatrixIR(ctx, x)
    case x: MatrixIR => LowerMatrixIR(ctx, x)
    case x: BlockMatrixIR => LowerMatrixIR(ctx, x)
  }
}

case object LiftRelationalValuesToRelationalLets extends LoweringPass {
  val before: Invariant = NoMatrixIR
  val after: Invariant = NoMatrixIR and NoLiftMeOuts
  val context: String = "LiftRelationalValuesToRelationalLets"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LiftRelationalValues(ir)
}

case object LegacyInterpretNonCompilablePass extends LoweringPass {
  val before: Invariant = NoMatrixIR
  val after: Invariant = NoMatrixIR and NoRelationalLets and CompilableValueIRs
  val context: String = "InterpretNonCompilable"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerOrInterpretNonCompilable(ctx, ir)
}

case object LowerOrInterpretNonCompilablePass extends LoweringPass {
  val before: Invariant = NoMatrixIR
  val after: Invariant = CompilableIR
  val context: String = "LowerOrInterpretNonCompilable"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerOrInterpretNonCompilable(ctx, ir)
}

case class LowerToDistributedArrayPass(t: DArrayLowering.Type) extends LoweringPass {
  val before: Invariant = NoMatrixIR
  val after: Invariant = CompilableIR
  val context: String = "LowerToDistributedArray"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerToCDA(ir.asInstanceOf[IR], t, ctx)
}

case object InlineApplyIR extends LoweringPass {
  val before: Invariant = CompilableIR
  val after: Invariant = CompilableIR and NoApplyIR
  val context: String = "InlineApplyIR"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      RewriteBottomUp(
        ir,
        {
          case x: ApplyIR => Some(x.explicitNode)
          case _ => None
        },
      )
    }
}

case object LowerArrayAggsToRunAggsPass extends LoweringPass {
  val before: Invariant = CompilableIR and NoApplyIR
  val after: Invariant = EmittableIR
  val context: String = "LowerArrayAggsToRunAggs"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      val x = ir.noSharing(ctx)
      val r = Requiredness(x, ctx)
      RewriteBottomUp(
        x,
        {
          case x @ StreamAgg(a, name, query) =>
            val aggs = Extract(ctx, query, r)

            val newNode = Let(
              aggs.initBindings,
              RunAgg(
                Begin(FastSeq(
                  aggs.init,
                  StreamFor(a, name, aggs.seqPerElt),
                )),
                aggs.result,
                aggs.sigs.states,
              ),
            )

            if (newNode.typ != x.typ)
              throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")
            Some(newNode.noSharing(ctx))
          case x @ StreamAggScan(a, name, query) =>
            val aggs = Extract(ctx, query, r, isScan = true)

            val newNode = Let(
              aggs.initBindings,
              RunAggScan(
                a,
                name,
                aggs.init,
                aggs.seqPerElt,
                aggs.result,
                aggs.sigs.states,
              ),
            )
            if (newNode.typ != x.typ)
              throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")
            Some(newNode.noSharing(ctx))
          case _ => None
        },
      )
    }
}

case class EvalRelationalLetsPass(passesBelow: LoweringPipeline) extends LoweringPass {
  val before: Invariant = NoMatrixIR and NoLiftMeOuts
  val after: Invariant = before and NoRelationalLets
  val context: String = "EvalRelationalLets"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    EvalRelationalLets(ir, ctx, passesBelow)
}

case class LowerAndExecuteShufflesPass(passesBelow: LoweringPipeline) extends LoweringPass {
  val before: Invariant = NoRelationalLets and NoMatrixIR
  val after: Invariant = before and LoweredShuffles
  val context: String = "LowerAndExecuteShuffles"

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerAndExecuteShuffles(ir, ctx, passesBelow)
}
