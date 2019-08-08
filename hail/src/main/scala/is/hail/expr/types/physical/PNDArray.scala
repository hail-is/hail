package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.expr.Nat
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TNDArray

final case class PNDArray(elementType: PType, nDims: Int, override val required: Boolean = false) extends ComplexPType {
  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims), required)
  assert(elementType.required, "elementType must be required")

  def _toPretty = s"NDArray[$elementType,$nDims]"

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = throw new UnsupportedOperationException

  val representation: PStruct = {
    PStruct(required,
      ("flags", PInt32Required),
      ("offset", PInt32Required),
      ("shape", PTuple(true, Array.tabulate(nDims)(_ => PInt32Required):_*)),
      ("strides", PArray(PInt32Required, required = true)),
      ("data", PArray(elementType, required = true)))
  }
}
