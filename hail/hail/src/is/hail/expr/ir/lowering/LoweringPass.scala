package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.agg.Extract
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.invariant._
import is.hail.utils.TimedBlock

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
  def context: String
  def before: Invariant
  def after: Invariant

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    TimedBlock.enter {
      before.verify(ctx, ir)
      val result = transform(ctx, ir)
      after.verify(ctx, result)
      result
    }

  protected def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR
}

case class OptimizePass(_context: String) extends LoweringPass {
  override val context = s"Optimize: ${_context}"
  override def before: Invariant = LowerableIR
  override def after: Invariant = LowerableIR

  override def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    if (ctx.flags.isDefined(Optimize.Flags.Optimize)) super.apply(ctx, ir)
    else ir

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = Optimize(ctx, ir)
}

case object LowerMatrixToTablePass extends LoweringPass {
  override val context: String = "LowerMatrixToTable"
  override def before: Invariant = LowerableIR
  override def after: Invariant = before and NoMatrixIR
  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LowerMatrixIR(ctx, ir)
}

case object LiftRelationalValuesToRelationalLets extends LoweringPass {
  override val context: String = "LiftRelationalValuesToRelationalLets"
  override def before: Invariant = LowerableIR and NoMatrixIR
  override def after: Invariant = before
  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LiftRelationalValues(ctx, ir)
}

case object LegacyInterpretNonCompilablePass extends LoweringPass {
  override val context: String = "InterpretNonCompilable"
  override def before: Invariant = LowerableIR and NoMatrixIR
  override def after: Invariant = before and NoRelationalLets and CompilableValueIRs

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerOrInterpretNonCompilable(ctx, ir)
}

case object LowerOrInterpretNonCompilablePass extends LoweringPass {
  override val context: String = "LowerOrInterpretNonCompilable"
  override def before: Invariant = LowerableIR and NoMatrixIR
  override def after: Invariant = LowerableIR and CompilableIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerOrInterpretNonCompilable(ctx, ir)
}

case class LowerToDistributedArrayPass(t: DArrayLowering.Type) extends LoweringPass {
  override val context: String = "LowerToDistributedArray"
  override def before: Invariant = LowerableIR and NoMatrixIR
  override def after: Invariant = LowerableIR and CompilableIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerToCDA(ir.asInstanceOf[IR], t, ctx)
}

case object InlineApplyIR extends LoweringPass {
  override val context: String = "InlineApplyIR"
  override def before: Invariant = LowerableIR and CompilableIR
  override def after: Invariant = before and NoApplyIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    TimedBlock.enter {
      val rewritten =
        RewriteBottomUp(
          ir,
          {
            case x: ApplyIR => Some(x.explicitNode)
            case _ => None
          },
        )

      if (rewritten ne ir) NormalizeNames()(ctx, rewritten)
      else ir
    }
}

case object LowerArrayAggsToRunAggsPass extends LoweringPass {
  override val context: String = "LowerArrayAggsToRunAggs"
  override def before: Invariant = LowerableIR and CompilableIR and NoApplyIR
  override def after: Invariant = LowerableIR and EmittableIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    TimedBlock.enter {
      val r = Requiredness(ir, ctx)

      val rewritten =
        RewriteBottomUp(
          ir,
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

              Some(newNode)
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
              Some(newNode)
            case _ => None
          },
        )

      if (rewritten ne ir) NormalizeNames()(ctx, rewritten)
      else ir
    }
}

case class EvalRelationalLetsPass(passesBelow: LoweringPipeline) extends LoweringPass {
  override val context: String = "EvalRelationalLets"
  override def before: Invariant = LowerableIR and NoMatrixIR
  override def after: Invariant = before and NoRelationalLets

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    EvalRelationalLets(ir, ctx, passesBelow)
}

case object LowerTableKeyByAndAggregatePass extends LoweringPass {
  override val context: String = "LowerTableKeyByAndAggregate"
  override def before: Invariant = LowerableIR and NoRelationalLets and NoMatrixIR
  override def after: Invariant = before and NoTableKeyByAndAggregate

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerTableKeyByAndAggregate(ctx, ir)
}

case object LowerAndExecuteShufflesPass extends LoweringPass {
  override val context: String = "LowerAndExecuteShuffles"

  override def before: Invariant =
    LowerableIR and NoRelationalLets and NoMatrixIR and NoTableKeyByAndAggregate

  override def after: Invariant =
    before and LoweredShuffles

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerAndExecuteShuffles(ir, ctx)
}
