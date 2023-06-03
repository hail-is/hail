package is.hail.expr.ir.lowering

import cats.syntax.all._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.utils.traverseInstanceGenTraversable

import scala.language.higherKinds

object LowerToCDA {

  def apply[M[_]: MonadLower](ir: IR, typesToLower: DArrayLowering.Type): M[IR] =
    for {
      analyses <- LoweringAnalyses(ir)
      result <- lower(ir, typesToLower, analyses)
    } yield result

  def lower[M[_]](ir: IR, typesToLower: DArrayLowering.Type, analyses: LoweringAnalyses)
                 (implicit M: MonadLower[M]): M[IR] =
    ir match {
      case node if node.children.forall(_.isInstanceOf[IR]) =>
        ir.children.traverse { case c: IR => lower(c, typesToLower, analyses) }.map(Copy(node, _))

      case node if node.children.exists(n => n.isInstanceOf[TableIR]) && node.children.forall(n => n.isInstanceOf[TableIR] || n.isInstanceOf[IR]) =>
        LowerTableIR(ir, typesToLower, analyses)

      case node if node.children.exists(n => n.isInstanceOf[BlockMatrixIR]) && node.children.forall(n => n.isInstanceOf[BlockMatrixIR] || n.isInstanceOf[IR]) =>
        LowerBlockMatrixIR(ir, typesToLower, analyses)

      case node if node.children.exists(_.isInstanceOf[MatrixIR]) =>
        M.raiseError(new LowererUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${Pretty(node)}"))

      case node =>
        M.raiseError(new LowererUnsupportedOperation(s"Cannot lower: \n${Pretty(node)}"))
    }
}

object DArrayLowering extends Enumeration {
  type Type = Value
  val All, TableOnly, BMOnly = Value
  def lowerTable(t: Type): Boolean = t == All || t == TableOnly
  def lowerBM(t: Type): Boolean = t == All || t == BMOnly
}
