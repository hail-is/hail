package is.hail.expr.types.encoded

abstract class EContainer extends EType

final case class EArray(elementType: EType, override val required: Boolean = false) extends EContainer

final case class ESet(elementType: EType, override val required: Boolean = false) extends EContainer

final case class EDict(keyType: EType, valueType: EType, override val required: Boolean = false) extends EContainer
