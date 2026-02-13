package is.hail.expr.ir

import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.defs.{Begin, EncodedLiteral, Literal, MakeTuple, NA}
import is.hail.expr.ir.lowering.{Invariant, LoweringPipeline}
import is.hail.types.physical.PTuple
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._

import java.io.PrintWriter

import org.apache.spark.sql.Row

object CompileAndEvaluate {
  def apply[T](
    ctx: ExecuteContext,
    ir: IR,
    lower: LoweringPipeline = LoweringPipeline(),
  ): T =
    ctx.time {
      _apply(ctx, ir, lower) match {
        case Left(()) => ().asInstanceOf[T]
        case Right((t, off)) => SafeRow(t, off).getAs[T](0)
      }
    }

  def evalToIR(ctx: ExecuteContext, ir: IR): IR =
    if (IsConstant(ir)) ir
    else _apply(ctx, ir) match {
      case Left(_) => Begin(FastSeq())
      case Right((pt, addr)) =>
        ir.typ match {
          case _ if pt.isFieldMissing(addr, 0) => NA(ir.typ)
          case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean | TString =>
            Literal.coerce(ir.typ, SafeRow.read(pt, addr).asInstanceOf[Row].get(0))
          case _ => EncodedLiteral.fromPTypeAndAddress(pt.types(0), pt.loadField(addr, 0), ctx)
        }
    }

  def _apply(
    ctx: ExecuteContext,
    ir0: IR,
    lower: LoweringPipeline = LoweringPipeline(),
    print: Option[PrintWriter] = None,
  ): Either[Unit, (PTuple, Long)] =
    ctx.time {
      val ir = lower(ctx, ir0).asInstanceOf[IR]
      Invariant.CompilableIR.verify(ctx, ir)

      ir.typ match {
        case TVoid =>
          val (_, f) = Compile[AsmFunction1RegionUnit](
            ctx,
            FastSeq(),
            FastSeq(classInfo[Region]),
            UnitInfo,
            ir,
            print,
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
              print,
            )

          val res = ctx.scopedExecution { (hcl, fs, htc, r) =>
            val execute = f(hcl, fs, htc, r)
            ctx.time(execute(r))
          }

          Right((resType, res))
      }
    }
}
