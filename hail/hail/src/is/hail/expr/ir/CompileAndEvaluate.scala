package is.hail.expr.ir

import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.defs.{Begin, EncodedLiteral, Literal, MakeTuple, NA}
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.types.physical.PTuple
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils.FastSeq

import org.apache.spark.sql.Row

object CompileAndEvaluate {
  def apply[T](ctx: ExecuteContext, ir0: IR, optimize: Boolean = true): T = {
    ctx.time {
      _apply(ctx, ir0, optimize) match {
        case Left(()) => ().asInstanceOf[T]
        case Right((t, off)) => SafeRow(t, off).getAs[T](0)
      }
    }
  }

  def evalToIR(ctx: ExecuteContext, ir0: IR, optimize: Boolean = true): IR = {
    if (IsConstant(ir0))
      return ir0

    _apply(ctx, ir0, optimize) match {
      case Left(_) => Begin(FastSeq())
      case Right((pt, addr)) =>
        ir0.typ match {
          case _ if pt.isFieldMissing(addr, 0) => NA(ir0.typ)
          case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean | TString =>
            Literal.coerce(ir0.typ, SafeRow.read(pt, addr).asInstanceOf[Row].get(0))
          case _ => EncodedLiteral.fromPTypeAndAddress(pt.types(0), pt.loadField(addr, 0), ctx)
        }
    }
  }

  def _apply(
    ctx: ExecuteContext,
    ir0: IR,
    optimize: Boolean = true,
  ): Either[Unit, (PTuple, Long)] =
    ctx.time {
      val ir = LoweringPipeline.relationalLowerer(optimize)(ctx, ir0).asInstanceOf[IR]

      ir.typ match {
        case TVoid =>
          val (_, f) = Compile[AsmFunction1RegionUnit](
            ctx,
            FastSeq(),
            FastSeq(classInfo[Region]),
            UnitInfo,
            ir,
            print = None,
            optimize = optimize,
          )

          val unit: Unit = ctx.scopedExecution { (hcl, fs, htc, r) =>
            val execute = f(hcl, fs, htc, r)
            ctx.time(execute(r))
          }

          Left(unit)

        case _ =>
          val (Some(PTypeReferenceSingleCodeType(resType: PTuple)), f) =
            Compile[AsmFunction1RegionLong](
              ctx,
              FastSeq(),
              FastSeq(classInfo[Region]),
              LongInfo,
              MakeTuple.ordered(FastSeq(ir)),
              print = None,
              optimize = optimize,
            )

          val res = ctx.scopedExecution { (hcl, fs, htc, r) =>
            val execute = f(hcl, fs, htc, r)
            ctx.time(execute(r))
          }

          Right((resType, res))
      }
    }
}
