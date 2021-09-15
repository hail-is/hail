package is.hail.types.physical

abstract class PIterable extends PType {
  def elementType: PType
}
