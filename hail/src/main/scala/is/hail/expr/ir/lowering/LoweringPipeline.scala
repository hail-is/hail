package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, IRSize, Pretty, TypeCheck}
import is.hail.utils._

case class LoweringPipeline(lowerings: LoweringPass*) {
  assert(lowerings.nonEmpty)

  final def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    var x = ir

    def render(context: String): Unit = {
      if (ctx.shouldLogIR())
        log.info(s"$context: IR size ${ IRSize(x) }: \n" + Pretty(ctx, x, elideLiterals = true))
    }

    render(s"initial IR")

    lowerings.foreach { l =>
      try {
        x = l.apply(ctx, x)
        render(s"after ${ l.context }")
      } catch {
        case e: Throwable =>
          log.error(s"error while applying lowering '${ l.context }'")
          throw e
      }
      try {
        TypeCheck(ctx, x)
      } catch {
        case e: Throwable =>
          fatal(s"error after applying ${ l.context }", e)
      }
    }

    x
  }

  def noOptimization(): LoweringPipeline = LoweringPipeline(lowerings.filter(l => !l.isInstanceOf[OptimizePass]): _*)

  def +(suffix: LoweringPipeline): LoweringPipeline = LoweringPipeline((lowerings ++ suffix.lowerings): _*)
}

object LoweringPipeline {

  def fullLoweringPipeline(context: String, baseTransformer: LoweringPass): LoweringPipeline = {

    val base = LoweringPipeline(
      baseTransformer,
      OptimizePass(s"$context, after ${ baseTransformer.context }")
    )

    // recursively lowers and executes
    val withShuffleRewrite =
      LoweringPipeline(
        LowerAndExecuteShufflesPass(base),
        OptimizePass(s"$context, after LowerAndExecuteShuffles")
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
      OptimizePass(s"$context, after LowerMatrixToTable")
    ) + withLetEvaluation
  }

  private val _relationalLowerer = fullLoweringPipeline("relationalLowerer", LowerOrInterpretNonCompilablePass)
  private val _relationalLowererNoOpt = _relationalLowerer.noOptimization()


  // legacy lowers can run partial optimization on a TableIR/MatrixIR that gets interpreted to a
  // TableValue for spark compatibility
  private val _relationalLowererLegacy = fullLoweringPipeline("relationalLowererLegacy", LegacyInterpretNonCompilablePass)
  private val _relationalLowererNoOptLegacy = _relationalLowererLegacy.noOptimization()

  private val _compileLowerer = LoweringPipeline(
    OptimizePass("compileLowerer, initial IR"),
    InlineApplyIR,
    OptimizePass("compileLowerer, after InlineApplyIR"),
    LowerArrayAggsToRunAggsPass,
    OptimizePass("compileLowerer, after LowerArrayAggsToRunAggs")
  )
  private val _compileLowererNoOpt = _compileLowerer.noOptimization()

  private val _dArrayLowerers: Map[DArrayLowering.Type, LoweringPipeline] =
    Array(
      DArrayLowering.All,
      DArrayLowering.TableOnly,
      DArrayLowering.BMOnly
    ).map { lv =>
      lv -> fullLoweringPipeline("darrayLowerer", LowerToDistributedArrayPass(lv))
    }.toMap

  private val _dArrayLowerersNoOpt = _dArrayLowerers.mapValues(_.noOptimization()).toMap

  def relationalLowerer(optimize: Boolean): LoweringPipeline = if (optimize) _relationalLowerer else _relationalLowererNoOpt

  def legacyRelationalLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize) _relationalLowererLegacy else _relationalLowererNoOptLegacy

  def darrayLowerer(optimize: Boolean): Map[DArrayLowering.Type, LoweringPipeline] = if (optimize) _dArrayLowerers else _dArrayLowerersNoOpt

  def compileLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize) _compileLowerer else _compileLowererNoOpt
}
