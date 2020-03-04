package is.hail.expr.ir.lowering

import is.hail.expr.ir.{BlockMatrixIR, Copy, IR, MatrixIR, Pretty, TableIR}

object LowerIR {
  def lower(ir: IR, typesToLower: DArrayLowering.Type): IR = ir match {
    case node if node.children.forall(n => n.isInstanceOf[TableIR] || n.isInstanceOf[IR] ) =>
      LowerTableIR.lower(ir, typesToLower)

    case node if node.children.exists( _.isInstanceOf[MatrixIR] ) =>
      throw new LowererUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${ Pretty(node) }")

    case node if node.children.forall(n => n.isInstanceOf[IR] || n.isInstanceOf[BlockMatrixIR] ) =>
      LowerBlockMatrixIR.lower(ir, typesToLower)

    case node if node.children.forall(_.isInstanceOf[IR]) =>
      Copy(node, ir.children.map { case c: IR => lower(c, typesToLower) })

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
