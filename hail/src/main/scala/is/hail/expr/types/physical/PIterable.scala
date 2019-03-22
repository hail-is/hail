package is.hail.expr.types.physical

abstract class PIterable extends PType {
  def elementType: PType

  def asPContainer: PContainer = this match {
    case _: PStream => PArray(this.elementType, this.required)
    case x: PContainer => x
  }
}
