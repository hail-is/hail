package is.hail.types.encoded

final case class EDictAsUnsortedArrayOfPairs(val elementType: EType, override val required: Boolean = false) extends EContainer {
}
