package is.hail.expr.ir
import is.hail.expr.types.physical.PType

object InferRequired {
  def apply(pType: PType) {

  }

  def apply(pTypes: PType*): Boolean = pTypes.forall(pType => pType.required)
}


