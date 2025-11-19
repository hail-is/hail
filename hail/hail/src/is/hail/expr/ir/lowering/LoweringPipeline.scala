package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, IRSize, Pretty, TypeCheck}
import is.hail.utils._

case class LoweringPipeline(lowerings: LoweringPass*) extends Logging {
  final def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    if (lowerings.isEmpty) ir
    else ctx.time {
      var x = ir

      def render(context: String): Unit =
        if (ctx.shouldLogIR())
          logger.info(s"$context: IR size ${IRSize(x)}: \n" + Pretty(ctx, x))

      render(s"initial IR")

      for (l <- lowerings) {
        try {
          x = l(ctx, x)
          render(s"after ${l.context}")
        } catch {
          case e: Throwable =>
            logger.error(s"error while applying lowering '${l.context}'", e)
            throw e
        }
        try
          TypeCheck(ctx, x)
        catch {
          case e: Throwable =>
            fatal(s"error after applying ${l.context}", e)
        }
      }

      x
    }
  }

  def +(suffix: LoweringPipeline): LoweringPipeline =
    LoweringPipeline((lowerings ++ suffix.lowerings): _*)
}

object LoweringPipeline {

  private[this] def fullLoweringPipeline(context: String, baseTransformer: LoweringPass)
    : LoweringPipeline = {

    val base = LoweringPipeline(
      baseTransformer,
      OptimizePass(s"$context, after ${baseTransformer.context}"),
    )

    // recursively lowers and executes
    val withShuffleRewrite =
      LoweringPipeline(
        LowerAndExecuteShufflesPass(base),
        OptimizePass(s"$context, after LowerAndExecuteShuffles"),
      ) + base

    // recursively lowers and executes
    val withLetEvaluation =
      LoweringPipeline(
        LiftRelationalValuesToRelationalLets,
        EvalRelationalLetsPass(withShuffleRewrite),
      ) + withShuffleRewrite

    LoweringPipeline(
      OptimizePass(s"$context, initial IR"),
      LowerMatrixToTablePass,
      OptimizePass(s"$context, after LowerMatrixToTable"),
    ) + withLetEvaluation
  }

  lazy val relationalLowerer: LoweringPipeline =
    fullLoweringPipeline("relationalLowerer", LowerOrInterpretNonCompilablePass)

  // legacy lowers can run partial optimization on a TableIR/MatrixIR that gets interpreted to a
  // TableValue for spark compatibility
  lazy val legacyRelationalLowerer: LoweringPipeline =
    fullLoweringPipeline("relationalLowererLegacy", LegacyInterpretNonCompilablePass)

  lazy val compileLowerer: LoweringPipeline =
    LoweringPipeline(
      OptimizePass("compileLowerer, initial IR"),
      InlineApplyIR,
      OptimizePass("compileLowerer, after InlineApplyIR"),
      LowerArrayAggsToRunAggsPass,
      OptimizePass("compileLowerer, after LowerArrayAggsToRunAggs"),
    )

  lazy val darrayLowerer: Map[DArrayLowering.Value, LoweringPipeline] =
    Array(
      DArrayLowering.All,
      DArrayLowering.TableOnly,
      DArrayLowering.BMOnly,
    )
      .map(lv => lv -> fullLoweringPipeline("darrayLowerer", LowerToDistributedArrayPass(lv)))
      .toMap

}
