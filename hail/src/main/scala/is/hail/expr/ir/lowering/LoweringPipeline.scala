package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BaseIR, ExecuteContext}
import is.hail.utils._

case class LoweringPipeline(lowerings: LoweringPass*) {
  assert(lowerings.nonEmpty)
  lowerings.zip(lowerings.tail).foreach { case (l, r) =>
    assert(l.after == r.before)
  }

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
}

object LoweringPipeline {
  def relationalLowerer(optimize: Boolean): LoweringPipeline = if (optimize)
    LoweringPipeline(
      OptimizePass("relationalLowerer, initial IR"),
      LowerMatrixToTablePass,
      OptimizePass("relationalLowerer, after LowerMatrixToTable"),
      InterpretNonCompilablePass,
      OptimizePass("relationalLowerer, after InterpretNonCompilable")
    )
  else
    LoweringPipeline(
      LowerMatrixToTablePass,
      InterpretNonCompilablePass
    )

  def legacyRelationalLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize)
      LoweringPipeline(
        OptimizePass("legacyRelationalLowerer, initial IR"),
        LowerMatrixToTablePass,
        OptimizePass("legacyRelationalLowerer, after LowerMatrixToTable"),
        LegacyInterpretNonCompilablePass,
        OptimizePass("legacyRelationalLowerer, after LegacyInterpretNonCompilable")
      )
    else
      LoweringPipeline(
        LowerMatrixToTablePass,
        LegacyInterpretNonCompilablePass
      )

  def compileLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize)
      LoweringPipeline(
        OptimizePass("compileLowerer, initial IR"),
        InlineApplyIR,
        OptimizePass("compileLowerer, after InlineApplyIR"),
        LowerArrayAggsToRunAggsPass,
        OptimizePass("compileLowerer, after LowerArrayAggsToRunAggs")
      )
    else

      LoweringPipeline(InlineApplyIR, LowerArrayAggsToRunAggsPass)

  def darrayLowerer(optimize: Boolean): Map[DArrayLowering.Type, LoweringPipeline] = if (optimize)
    Map(
      DArrayLowering.All -> LoweringPipeline(
        OptimizePass("darrayLowerer, initial IR"),
        LowerMatrixToTablePass,
        OptimizePass("darrayLowerer, after LowerMatrixToTable"),
        LowerToDistributedArrayPass(DArrayLowering.All),
        OptimizePass("darrayLowerer, after LowerToCDA")
      ),
      DArrayLowering.TableOnly -> LoweringPipeline(
        OptimizePass("darrayLowerer, initial IR"),
        LowerMatrixToTablePass,
        OptimizePass("darrayLowerer, after LowerMatrixToTable"),
        LowerToDistributedArrayPass(DArrayLowering.TableOnly),
        OptimizePass("darrayLowerer, after LowerToCDA")
      ),
      DArrayLowering.BMOnly -> LoweringPipeline(
        OptimizePass("darrayLowerer, initial IR"),
        LowerMatrixToTablePass,
        OptimizePass("darrayLowerer, after LowerMatrixToTable"),
        LowerToDistributedArrayPass(DArrayLowering.BMOnly),
        OptimizePass("darrayLowerer, after LowerToCDA")
      ))
  else
    Map(
      DArrayLowering.All -> LoweringPipeline(
        LowerMatrixToTablePass,
        LowerToDistributedArrayPass(DArrayLowering.All)
      ),
      DArrayLowering.TableOnly -> LoweringPipeline(
        LowerMatrixToTablePass,
        LowerToDistributedArrayPass(DArrayLowering.TableOnly)
      ),
      DArrayLowering.BMOnly -> LoweringPipeline(
        LowerMatrixToTablePass,
        LowerToDistributedArrayPass(DArrayLowering.BMOnly)
      ))
}
