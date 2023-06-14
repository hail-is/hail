package is.hail.expr.ir.lowering

import cats.syntax.all._
import is.hail.expr.ir.{BaseIR, IRSize, Pretty, TypeCheck}
import is.hail.utils._

import scala.language.higherKinds
import scala.util.control.NonFatal

case class LoweringPipeline(lowerings: LoweringPass*) {
  assert(lowerings.nonEmpty)

  final def apply[M[_]](ir0: BaseIR)(implicit M: MonadLower[M]): M[BaseIR] = {
    def render(context: String, ir: BaseIR): M[Unit] =
      M.reader { ctx =>
        if (ctx.shouldLogIR())
          log.info(s"$context: IR size ${IRSize(ir)}: \n" + Pretty(ctx, ir))
      }

    render(s"initial IR", ir0) *> FastSeq(lowerings: _*).foldM(ir0) { case (ir, lower) =>
      for {
        lowered <- lower(ir).handleErrorWith {
          case NonFatal(e) =>
            log.error(s"error while applying lowering '${lower.context}'")
            M.raiseError(e)
        }
        _ <- render(s"after ${lower.context}", lowered)
        _ <- TypeCheck(lowered).handleErrorWith {
          case NonFatal(e) =>
            M.raiseError(new HailException(s"error after applying ${lower.context}", None, e))
        }
      } yield lowered
    }
  }

  def noOptimization(): LoweringPipeline =
    LoweringPipeline(lowerings.filter(l => !l.isInstanceOf[OptimizePass]): _*)

  def +(suffix: LoweringPipeline): LoweringPipeline =
    LoweringPipeline((lowerings ++ suffix.lowerings): _*)
}

object LoweringPipeline {

  def fullLoweringPipeline(context: String, baseTransformer: LoweringPass): LoweringPipeline = {

    val base =
      LoweringPipeline(
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
      ComputeSemanticHash,
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

  private val _dArrayLowerers = Array(
    DArrayLowering.All,
    DArrayLowering.TableOnly,
    DArrayLowering.BMOnly).map { lv =>
    (lv -> fullLoweringPipeline("darrayLowerer", LowerToDistributedArrayPass(lv)))
  }.toMap

  private val _dArrayLowerersNoOpt = _dArrayLowerers.mapValues(_.noOptimization()).toMap

  def relationalLowerer(optimize: Boolean): LoweringPipeline = if (optimize) _relationalLowerer else _relationalLowererNoOpt

  def legacyRelationalLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize) _relationalLowererLegacy else _relationalLowererNoOptLegacy

  def darrayLowerer(optimize: Boolean): Map[DArrayLowering.Type, LoweringPipeline] = if (optimize) _dArrayLowerers else _dArrayLowerersNoOpt

  def compileLowerer(optimize: Boolean): LoweringPipeline =
    if (optimize) _compileLowerer else _compileLowererNoOpt
}
