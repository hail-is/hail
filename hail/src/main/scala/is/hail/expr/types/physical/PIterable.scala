package is.hail.expr.types.physical

abstract class PIterable extends PType {
  def elementType: PType
}
