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
    val analyses = LoweringAnalyses.apply(ir, ctx)

    lower(ir, typesToLower, ctx, analyses)
  }

  def lower(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): IR = ir match {
    case node if node.children.forall(_.isInstanceOf[IR]) =>
      ir.mapChildren { case c: IR => lower(c, typesToLower, ctx, analyses) }

    case node if node.children.exists(n => n.isInstanceOf[TableIR]) && node.children.forall(n => n.isInstanceOf[TableIR] || n.isInstanceOf[IR]) =>
      LowerTableIR(ir, typesToLower, ctx, analyses)

    case node if node.children.exists(n => n.isInstanceOf[BlockMatrixIR]) && node.children.forall(n => n.isInstanceOf[BlockMatrixIR] || n.isInstanceOf[IR]) =>
      LowerBlockMatrixIR(ir, typesToLower, ctx, analyses)

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
