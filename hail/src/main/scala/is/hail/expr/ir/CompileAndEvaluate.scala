package is.hail.expr.ir

import cats.syntax.all._
import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s._
import is.hail.backend.utils._
import is.hail.expr.ir.lowering.{LoweringPipeline, MonadLower}
import is.hail.types.physical.PTuple
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.apache.spark.sql.Row

import scala.language.higherKinds

object CompileAndEvaluate {
  def apply[M[_], T](ir0: IR, optimize: Boolean = true)
                    (implicit M: MonadLower[M]): M[T] =
    timeM("CompileAndEvaluate") {
      _apply(ir0, optimize).map {
        case Left(()) => ().asInstanceOf[T]
        case Right((t, off)) => SafeRow(t, off).getAs[T](0)
      }
    }

  def evalToIR[M[_]](ir0: IR, optimize: Boolean = true)(implicit M: MonadLower[M]): M[IR] =
    if (IsConstant(ir0)) M.pure(ir0)
    else M.product(_apply(ir0, optimize), M.ask).map {
      case (Left(_), _) => Begin(FastIndexedSeq())
      case (Right((pt, addr)), ctx) =>
        ir0.typ match {
          case _ if pt.isFieldMissing(addr, 0) => NA(ir0.typ)
          case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean | TString =>
            Literal.coerce(ir0.typ, SafeRow.read(pt, addr).asInstanceOf[Row].get(0))
          case _ => EncodedLiteral.fromPTypeAndAddress(pt.types(0), pt.loadField(addr, 0), ctx)
        }
    }

  def _apply[M[_]](ir0: IR, optimize: Boolean = true)(implicit M: MonadLower[M])
  : M[Either[Unit, (PTuple, Long)]] =
    LoweringPipeline.relationalLowerer(optimize)(ir0).flatMap { case ir: IR =>
      ir.typ match {
        case TVoid =>
          for {
            (_, f) <- timeM("Compile") {
              Compile[M, AsmFunction1RegionUnit](
                FastIndexedSeq(),
                FastIndexedSeq(classInfo[Region]), UnitInfo,
                ir,
                print = None,
                optimize = optimize
              )
            }

            _ <- scopedExecution { case (hcl, fs, htc, r) =>
              for {
                fRunnable <- time("InitializeCompiledFunction")(f(hcl, fs, htc, r))
                _ <- time("RunCompiledVoidFunction")(fRunnable(r))
              } yield ()
            }

          } yield Left(())

        case _ =>
          for {
            (Some(PTypeReferenceSingleCodeType(resType: PTuple)), f) <- timeM("Compile") {
              Compile[M, AsmFunction1RegionLong](
                FastIndexedSeq(),
                FastIndexedSeq(classInfo[Region]), LongInfo,
                MakeTuple.ordered(FastSeq(ir)),
                print = None,
                optimize = optimize
              )
            }

            ctx <- M.ask
            fRunnable <- time("InitializeCompiledFunction")(f(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r))
            resultAddress <- time("RunCompiledFunction")(fRunnable(ctx.r))

          } yield Right((resType, resultAddress))
      }
    }
}
