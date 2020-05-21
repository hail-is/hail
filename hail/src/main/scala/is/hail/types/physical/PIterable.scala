package is.hail.types.physical

abstract class PIterable extends PType {
  def elementType: PType

  def asPContainer: PContainer = this match {
    case _: PStream => PCanonicalArray(this.elementType, this.required)
    case x: PContainer => x
  }
}
