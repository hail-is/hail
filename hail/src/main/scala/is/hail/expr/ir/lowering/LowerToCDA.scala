package is.hail.expr.ir.lowering

import is.hail.annotations.{Region, SafeRow, UnsafeRow}
import is.hail.asm4s.{AsmFunction1RegionLong, AsmFunction1RegionUnit, LongInfo, UnitInfo, classInfo}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.physical.{PTuple, PType}
import is.hail.types.virtual.Type
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.apache.spark.sql.Row

object LowerToCDA {

  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext): IR = {
    val analyses = Analyses.apply(ir, ctx)

    lower(ir, typesToLower, ctx, analyses, Map())
  }

  def substLets(ir: IR, relationalLetsAbove: Map[String, IR]): IR = {
    RewriteBottomUp(ir, {
      case RelationalRef(name, t) =>
        relationalLetsAbove.get(name) match {
          case Some(lit) =>
            if (lit.typ != t)
              throw new RuntimeException(s"relational ref type mismatch:\n  ref: $t\n  lit: ${lit.typ}")
            Some(lit)
          case None => throw new RuntimeException(s"$ir had no match in relationalLetsAbove")
        }
      case _ => None
    }).asInstanceOf[IR]
  }

  def lower(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: Analyses, relationalLetsAbove: Map[String, IR]): IR = ir match {
    case RelationalLet(name, value, body) =>

      val loweredValue = substLets(lower(value, typesToLower, ctx, analyses, relationalLetsAbove), relationalLetsAbove)

      if (!Compilable(loweredValue))
        throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${ Pretty(ctx, ir) }")

      val (Some(PTypeReferenceSingleCodeType(pt: PTuple)), f) = ctx.timer.time("Compile") {
        Compile[AsmFunction1RegionLong](ctx,
          FastIndexedSeq(),
          FastIndexedSeq(classInfo[Region]), LongInfo,
          MakeTuple.ordered(FastSeq(loweredValue)),
          print = None)
      }

      val lit = ctx.scopedExecution { (hcl, fs, htc, r) =>

        val addr = ctx.timer.time("Run")(f(hcl, fs, htc, r).apply(r))

        if (pt.isFieldMissing(addr, 0))
          NA(pt.types(0).virtualType)
        else
          EncodedLiteral.fromPTypeAndAddress(pt.types(0), pt.loadField(addr, 0), ctx)
      }

      lower(body, typesToLower, ctx, analyses, relationalLetsAbove + ((name, lit)))

    case RelationalRef(name, t) =>
      relationalLetsAbove.get(name) match {
        case Some(lit) =>
          if (lit.typ != t)
            throw new RuntimeException(s"relational ref type mismatch:\n  ref: $t\n  lit: ${lit.typ}")
          lit
        case None => throw new RuntimeException(s"$ir had no match in relationalLetsAbove")
      }

    case node if node.children.forall(_.isInstanceOf[IR]) =>
      Copy(node, ir.children.map { case c: IR => lower(c, typesToLower, ctx, analyses, relationalLetsAbove) })

    case node if node.children.exists(n => n.isInstanceOf[TableIR]) && node.children.forall(n => n.isInstanceOf[TableIR] || n.isInstanceOf[IR]) =>
      LowerTableIR(ir, typesToLower, ctx, analyses, relationalLetsAbove)

    case node if node.children.exists(n => n.isInstanceOf[BlockMatrixIR]) && node.children.forall(n => n.isInstanceOf[BlockMatrixIR] || n.isInstanceOf[IR]) =>
      LowerBlockMatrixIR(ir, typesToLower, ctx, analyses, relationalLetsAbove)

    case node if node.children.exists(_.isInstanceOf[MatrixIR]) =>
      throw new LowererUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${ Pretty(ctx, node) }")

    case node =>
      throw new LowererUnsupportedOperation(s"Cannot lower: \n${ Pretty(ctx, node) }")
  }
}

object DArrayLowering extends Enumeration {
  type Type = Value
  val All, TableOnly, BMOnly = Value
  def lowerTable(t: Type): Boolean = t == All || t == TableOnly
  def lowerBM(t: Type): Boolean = t == All || t == BMOnly
}
