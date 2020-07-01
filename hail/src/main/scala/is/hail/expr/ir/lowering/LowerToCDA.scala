package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.types.virtual.Type

object LowerToCDA {

  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext): IR = {
    val r = Requiredness(ir, ctx)

    // Slightly simpler to replace RelationalRefs in a second linear pass,
    // since we'd have to embed the rewrite in many places otherwise
    RewriteBottomUp(lower(ir, typesToLower, ctx, r, Seq()), {
      case rr: RelationalRef => Some(Ref(rr.name, rr.typ))
      case _ => None
    }).asInstanceOf[IR]
  }

  def lower(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, r: RequirednessAnalysis, relationalLetsAbove: Seq[(String, Type)]): IR = ir match {
    case RelationalLet(name, value, body) =>
      Let(name, lower(value, typesToLower, ctx, r, relationalLetsAbove), lower(body, typesToLower, ctx, r, relationalLetsAbove :+ ((name, value.typ))))
    case node if node.children.forall(_.isInstanceOf[IR]) =>
      Copy(node, ir.children.map { case c: IR => lower(c, typesToLower, ctx, r, relationalLetsAbove) })

    case node if node.children.exists(n => n.isInstanceOf[TableIR]) && node.children.forall(n => n.isInstanceOf[TableIR] || n.isInstanceOf[IR]) =>
      LowerTableIR(ir, typesToLower, ctx, r, relationalLetsAbove)

    case node if node.children.exists(n => n.isInstanceOf[BlockMatrixIR]) && node.children.forall(n => n.isInstanceOf[BlockMatrixIR] || n.isInstanceOf[IR]) =>
      LowerBlockMatrixIR(ir, typesToLower, ctx, r, relationalLetsAbove)

    case node if node.children.exists(_.isInstanceOf[MatrixIR]) =>
      throw new LowererUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${ Pretty(node) }")

    case node =>
      throw new LowererUnsupportedOperation(s"Cannot lower: \n${ Pretty(node) }")
  }
}

object DArrayLowering extends Enumeration {
  type Type = Value
  val All, TableOnly, BMOnly = Value
  def lowerTable(t: Type): Boolean = t == All || t == TableOnly
  def lowerBM(t: Type): Boolean = t == All || t == BMOnly
}
