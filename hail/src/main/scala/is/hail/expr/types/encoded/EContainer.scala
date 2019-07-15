package is.hail.expr.types.encoded

import is.hail.expr.types.physical._

abstract class EContainer extends EType

final case class EArray(elementType: EType, override val required: Boolean = false) extends EContainer {
  def toPType(): PType = PArray(elementType.toPType(), required)
}

final case class ESet(elementType: EType, override val required: Boolean = false) extends EContainer {
  def toPType(): PType = PSet(elementType.toPType(), required)
}

final case class EDict(keyType: EType, valueType: EType, override val required: Boolean = false) extends EContainer {
  def toPType(): PType = PDict(keyType.toPType(), valueType.toPType(), required)
}
