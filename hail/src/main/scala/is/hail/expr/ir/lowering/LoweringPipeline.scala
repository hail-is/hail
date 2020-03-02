package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BaseIR, ExecuteContext, Optimize}
import is.hail.utils.HailException

case class LoweringPipeline(lowerings: IndexedSeq[LoweringPass]) {
  assert(lowerings.nonEmpty)
  lowerings.zip(lowerings.tail).foreach { case (l, r) =>
    assert(l.after == r.before)
  }

  final def apply(ctx: ExecuteContext, ir: BaseIR, optimize: Boolean): BaseIR = {
    var x = ir

    if (optimize)
      x = Optimize(x, noisy = true, context = "Lowerer, initial IR", ctx)

    lowerings.foreach { l =>
      try {
        x = l.apply(ctx, x)
        if (optimize)
          x = Optimize(x, noisy = true, context = s"${l.context}, post Lowering", ctx)
      } catch {
        case e: HailException => throw e
        case e: Throwable =>
          throw new RuntimeException(s"error while applying lowering '${ l.context }'", e)
      }
    }

    x
  }
}

object LoweringPipeline {
  val relationalLowerer: LoweringPipeline = LoweringPipeline(Array(LowerMatrixToTablePass, InterpretNonCompilablePass))
  val legacyRelationalLowerer: LoweringPipeline = LoweringPipeline(Array(LowerMatrixToTablePass, LegacyInterpretNonCompilablePass))
  val tableLowerer: LoweringPipeline = LoweringPipeline(Array(LowerMatrixToTablePass, LowerTableToDistributedArrayPass))
  val compileLowerer: LoweringPipeline = LoweringPipeline(Array(InlineApplyIR, LowerArrayAggsToRunAggsPass))
}
