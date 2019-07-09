package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.expr.Nat
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TNDArray

object PNDArray {
  def matMulNDims(l: Int, r: Int): Int = {
    (l, r) match {
      case (1, 1) => 0
      case (1, n) => n - 1
      case (n, 1) => n - 1
      case (_, _) => l
    }
  }
}

final case class PNDArray(elementType: PType, nDims: Int, override val required: Boolean = false) extends PType {
  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims), required)

  def _toPretty = s"NDArray[$elementType,$nDims]"

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = throw new UnsupportedOperationException
}
