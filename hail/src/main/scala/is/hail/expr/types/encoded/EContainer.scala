package is.hail.expr.types.encoded

abstract class EContainer extends EType {
  def elementType: EType
}
