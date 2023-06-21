package is.hail.expr.ir.lowering

import cats.Id
import cats.syntax.all._
import utils._
import is.hail.expr.ir._
import is.hail.expr.ir.agg.Extract
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.utils.{time => _, _}

import scala.language.higherKinds
case class LoweringState(semhash: Option[SemanticHash.Type] = None
                        )

object LoweringState {
  def nextHash[M[_]](implicit M: MonadLower[M]): M[Option[SemanticHash.Type]] =
    for {
      s <- M.get
      (h1, h2) = s.semhash match {
        case Some(hash) =>
          val (h1, h2) = SemanticHash.split(hash);
          (Some(h1), Some(h2))
        case None =>
          (None, None)
      }
      _ <- M.set(s.copy(semhash = h2))
    } yield h1
}

trait LoweringPass {
  val before: IRState
  val after: IRState
  val context: String

  final def apply[M[_]](ir: BaseIR)(implicit M: MonadLower[M]): M[BaseIR] =
    timeM(context) {
      for {
        _ <- time("Verify")(before.verify(ir))
        lowered <- timeM("LoweringTransformation")(transform(ir))
        _ <- time("Verify")(after.verify(lowered))
      } yield lowered
    }

  protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR]
}

case class OptimizePass(_context: String) extends LoweringPass {
  val context = s"optimize: ${_context}"
  val before: IRState = AnyIR
  val after: IRState = AnyIR

  override protected def transform[M[_]](ir: BaseIR)(implicit M: MonadLower[M]): M[BaseIR] =
    M.ask.flatMap { ctx =>
      unsafe { Optimize(ir, context, ctx) }
    }
}

case object LowerMatrixToTablePass extends LoweringPass {
  val before: IRState = AnyIR
  val after: IRState = MatrixLoweredToTable
  val context: String = "LowerMatrixToTable"

  override protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR] =
    LowerMatrixIR(ir)
}

case object LiftRelationalValuesToRelationalLets extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = MatrixLoweredToTable
  val context: String = "LiftRelationalValuesToRelationalLets"

  override protected def transform[M[_]](ir: BaseIR)(implicit M: MonadLower[M]): M[BaseIR] =
    M.pure { LiftRelationalValues(ir) }
}

case object LegacyInterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = ExecutableTableIR
  val context: String = "InterpretNonCompilable"

  override protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR] =
    LowerOrInterpretNonCompilable(ir)
}

case object LowerOrInterpretNonCompilablePass extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "LowerOrInterpretNonCompilable"

  override protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR] =
    LowerOrInterpretNonCompilable(ir)
}

case class LowerToDistributedArrayPass(t: DArrayLowering.Type) extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = CompilableIR
  val context: String = "LowerToDistributedArray"

  override protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR] =
    LowerToCDA(ir.asInstanceOf[IR], t).asInstanceOf[M[BaseIR]]
}

case object InlineApplyIR extends LoweringPass {
  val before: IRState = CompilableIR
  val after: IRState = CompilableIRNoApply
  val context: String = "InlineApplyIR"

  override protected def transform[M[_]](ir: BaseIR)(implicit M: MonadLower[M]): M[BaseIR] =
    unsafe[M, BaseIR] {
      RewriteBottomUp[Id](ir, {
        case x: ApplyIR => Some(x.explicitNode)
        case _ => None
      })
    }
}

case object LowerArrayAggsToRunAggsPass extends LoweringPass {
  val before: IRState = CompilableIRNoApply
  val after: IRState = EmittableIR
  val context: String = "LowerArrayAggsToRunAggs"

  override protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR] = {
      val x = ir.noSharing
      Requiredness(x).flatMap { r =>
        unsafe {
          RewriteBottomUp[Id](x, {
            case x@StreamAgg(a, name, query) =>
              val res = genUID()
              val aggs = Extract(query, res, r)

              val newNode = aggs.rewriteFromInitBindingRoot { root =>
                Let(
                  res,
                  RunAgg(
                    Begin(FastSeq(
                      aggs.init,
                      StreamFor(a, name, aggs.seqPerElt)
                    )),
                    aggs.results,
                    aggs.states
                  ),
                  root
                )
              }

              if (newNode.typ != x.typ)
                throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")

              Some(newNode.noSharing)

            case x@StreamAggScan(a, name, query) =>
              val res = genUID()
              val aggs = Extract(query, res, r, isScan = true)
              val newNode = aggs.rewriteFromInitBindingRoot { root =>
                RunAggScan(
                  a,
                  name,
                  aggs.init,
                  aggs.seqPerElt,
                  Let(res, aggs.results, root),
                  aggs.states
                )
              }
              if (newNode.typ != x.typ)
                throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")

              Some(newNode.noSharing)

            case _ => None
          })
        }
      }
  }
}

case class EvalRelationalLetsPass(passesBelow: LoweringPipeline) extends LoweringPass {
  val before: IRState = MatrixLoweredToTable
  val after: IRState = before + NoRelationalLetsState
  val context: String = "EvalRelationalLets"

  override protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR] =
    EvalRelationalLets(ir, passesBelow)
}

case class LowerAndExecuteShufflesPass(passesBelow: LoweringPipeline) extends LoweringPass {
  val before: IRState = NoRelationalLetsState + MatrixLoweredToTable
  val after: IRState = before + LoweredShuffles
  val context: String = "LowerAndExecuteShuffles"

  override protected def transform[M[_]: MonadLower](ir: BaseIR): M[BaseIR] =
    LowerAndExecuteShuffles(ir, passesBelow)

}

case object ComputeSemanticHash extends LoweringPass {
  override val before: IRState = AnyIR
  override val after: IRState = AnyIR
  override val context: String = "ComputeSemanticHash"

  override protected def transform[M[_]](ir: BaseIR)(implicit M: MonadLower[M]): M[BaseIR] =
    for {
      fs <- M.reader(_.fs)
      _ <- M.modify(_.copy(semhash = Some(SemanticHash(fs)(ir))))
    }  yield ir

}