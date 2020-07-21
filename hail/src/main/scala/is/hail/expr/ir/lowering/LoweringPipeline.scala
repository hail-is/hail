package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BaseIR, ExecuteContext}
import is.hail.utils._

case class LoweringPipeline(lowerings: LoweringPass*) {
  assert(lowerings.nonEmpty)

  final def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    var x = ir

    lowerings.foreach { l =>
      try {
        x = l.apply(ctx, x)
      } catch {
        case e: Throwable =>
          log.error(s"error while applying lowering '${ l.context }'")
          throw e
      }
    }

    x
  }

  def noOptimization(): LoweringPipeline = LoweringPipeline(lowerings.filter(l => !l.isInstanceOf[OptimizePass]): _*)
}

object LoweringPipeline {
  private val _relationalLowerer = LoweringPipeline(
    OptimizePass("relationalLowerer, initial IR"),
    LowerMatrixToTablePass,
    OptimizePass("relationalLowerer, after LowerMatrixToTable"),
    InterpretNonCompilablePass,
    OptimizePass("relationalLowerer, after InterpretNonCompilable"))
  private val _relationalLowererNoOpt = _relationalLowerer.noOptimization()


  private val _legacyRelationalLowerer = LoweringPipeline(
    OptimizePass("legacyRelationalLowerer, initial IR"),
    LowerMatrixToTablePass,
    OptimizePass("legacyRelationalLowerer, after LowerMatrixToTable"),
    LegacyInterpretNonCompilablePass,
    OptimizePass("legacyRelationalLowerer, after LegacyInterpretNonCompilable")
  )

  private val _legacyRelationalLowererNoOpt = _legacyRelationalLowerer.noOptimization()

  private val _compileLowerer = LoweringPipeline(
    OptimizePass("compileLowerer, initial IR"),
    InlineApplyIR,
    OptimizePass("compileLowerer, after InlineApplyIR"),
    LowerArrayAggsToRunAggsPass,
    OptimizePass("compileLowerer, after LowerArrayAggsToRunAggs")
  )
  private val _compileLowererNoOpt = _compileLowerer.noOptimization()

  private val _dArrayLowerers = Map(
    DArrayLowering.All -> LoweringPipeline(
      OptimizePass("darrayLowerer, initial IR"),
      LowerMatrixToTablePass,
      OptimizePass("darrayLowerer, after LowerMatrixToTable"),
      LiftRelationalValuesToRelationalLets,
      LowerToDistributedArrayPass(DArrayLowering.All),
      OptimizePass("darrayLowerer, after LowerToCDA")
    ),
    DArrayLowering.TableOnly -> LoweringPipeline(
      OptimizePass("darrayLowerer, initial IR"),
      LowerMatrixToTablePass,
      OptimizePass("darrayLowerer, after LowerMatrixToTable"),
      LiftRelationalValuesToRelationalLets,
      LowerToDistributedArrayPass(DArrayLowering.TableOnly),
      OptimizePass("darrayLowerer, after LowerToCDA")
    ),
    DArrayLowering.BMOnly -> LoweringPipeline(
      OptimizePass("darrayLowerer, initial IR"),
      LowerMatrixToTablePass,
      OptimizePass("darrayLowerer, after LowerMatrixToTable"),
      LiftRelationalValuesToRelationalLets,
      LowerToDistributedArrayPass(DArrayLowering.BMOnly),
      OptimizePass("darrayLowerer, after LowerToCDA")
    ))
  private val _dArrayLowerersNoOpt = _dArrayLowerers.mapValues(_.noOptimization()).toMap

  def relationalLowerer(optimize: Boolean): LoweringPipeline = if (optimize) _relationalLowerer else _relationalLowererNoOpt

  def legacyRelationalLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize) _legacyRelationalLowerer else _legacyRelationalLowererNoOpt

  def darrayLowerer(optimize: Boolean): Map[DArrayLowering.Type, LoweringPipeline] = if (optimize) _dArrayLowerers else _dArrayLowerersNoOpt

  def compileLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize) _compileLowerer else _compileLowererNoOpt
}
