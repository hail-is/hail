package is.hail.expr.ir.lowering

import is.hail.expr.ir.lowering.LowerTableIR.lower
import is.hail.expr.ir.{ArrayFlatMap, ArrayLen, BlockMatrixIR, Cast, Copy, GetField, IR, MakeStruct, MatrixIR, Pretty, Ref, TableCollect, TableCount, TableGetGlobals, TableIR, genUID, invoke}
import is.hail.expr.types.virtual.{TContainer, TInt64}
import is.hail.utils.FastIndexedSeq

object LowerIR {
  def lower(ir: IR, lowerTable: Boolean, lowerBM: Boolean): IR = ir match {
    case node if lowerTable && node.children.exists( _.isInstanceOf[TableIR] ) =>
      LowerTableIR.lower(ir)

    case node if node.children.exists( _.isInstanceOf[MatrixIR] ) =>
      throw new LowererUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${ Pretty(node) }")

    case node if lowerBM && node.children.exists( _.isInstanceOf[BlockMatrixIR] ) =>
      LowerBlockMatrixIR.lower(ir)

    case node if node.children.forall(_.isInstanceOf[IR]) =>
      Copy(node, ir.children.map { case c: IR => lower(c, lowerTable, lowerBM) })

    case node =>
      throw new LowererUnsupportedOperation(s"Cannot lower: \n${ Pretty(node) }")
  }
}
