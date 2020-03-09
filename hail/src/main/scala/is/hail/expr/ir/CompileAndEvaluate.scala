package is.hail.expr.ir

import is.hail.annotations.SafeRow
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.types.physical.{PTuple, PBaseStruct}
import is.hail.expr.types.virtual.TVoid
import is.hail.utils.FastSeq

object CompileAndEvaluate {
  def apply[T](ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean = true
  ): T = {
    ctx.timer.time("CompileAndEvaluate") {
      _apply(ctx, ir0, optimize) match {
        case Left(()) => ().asInstanceOf[T]
        case Right((t, off)) => SafeRow(t, off).getAs[T](0)
      }
    }
  }

  def _apply(
    ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean = true
  ): Either[Unit, (PTuple, Long)] = {
    val ir = LoweringPipeline.relationalLowerer.apply(ctx, ir0, optimize).asInstanceOf[IR]

    if (ir.typ == TVoid)
      // void is not really supported by IR utilities
      return Left(())

    val (resultPType: PTuple, f) = ctx.timer.time("Compile")(Compile[Long](ctx,
      MakeTuple.ordered(FastSeq(ir)), None, optimize = false))

    val fRunnable = ctx.timer.time("InitializeCompiledFunction")(f(0, ctx.r))
    val resultAddress = ctx.timer.time("RunCompiledFunction")(fRunnable(ctx.r))

    Right((resultPType, resultAddress))
  }
}
