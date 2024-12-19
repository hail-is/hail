package is.hail.types.encoded

abstract class EContainer extends EType {
  def elementType: EType
}
