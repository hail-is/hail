package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BaseIR, ExecuteContext, Optimize}

case class LoweringPipeline(lowerings: IndexedSeq[LoweringPass]) {
  assert(lowerings.nonEmpty)
  lowerings.zip(lowerings.tail).foreach { case (l, r) =>
    assert(l.after == r.before)
  }

  final def apply(ctx: ExecuteContext, ir: BaseIR, optimize: Boolean): BaseIR = {
    var x = ir

    if (optimize)
      x = Optimize.optimize(x, noisy = true, context = Some(s"initial optimization"))

    lowerings.foreach { l =>

      x = l.apply(ctx, x)

      if (optimize)
        x = Optimize.optimize(x, noisy = true, context = Some(s"${ l.context }, after"))
    }

    x
  }
}

object LoweringPipeline {
  val relationalLowerer: LoweringPipeline = LoweringPipeline(Array(LowerMatrixToTablePass, InterpretNonCompilablePass))
  val legacyRelationalLowerer: LoweringPipeline = LoweringPipeline(Array(LowerMatrixToTablePass, LegacyInterpretNonCompilablePass))
}