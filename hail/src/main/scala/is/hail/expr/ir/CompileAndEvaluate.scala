package is.hail.expr.ir

import is.hail.annotations.SafeRow
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.types.physical.PBaseStruct
import is.hail.expr.types.virtual.TVoid
import is.hail.utils.FastSeq

object CompileAndEvaluate {
  def apply[T](ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean = true
  ): T = {
    val ir = LoweringPipeline.relationalLowerer.apply(ctx, ir0, optimize).asInstanceOf[IR]

    if (ir.typ == TVoid)
    // void is not really supported by IR utilities
      return ().asInstanceOf[T]

    val (resultPType, f) = ctx.timer.time(Compile[Long](
      MakeTuple.ordered(FastSeq(ir)), None, optimize = false),
      "CompileAndEvaluate.compile")

    val fRunnable = ctx.timer.time(f(0, ctx.r), "create compiled function")
    val resultAddress = ctx.timer.time(fRunnable(ctx.r), "run compiled function")

    ctx.timer.time(
      SafeRow(resultPType.asInstanceOf[PBaseStruct], ctx.r, resultAddress).getAs[T](0),
      "convert to safe value")
  }
}
