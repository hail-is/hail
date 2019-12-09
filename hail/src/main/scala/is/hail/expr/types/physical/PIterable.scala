package is.hail.expr.types.physical

trait PIterable extends PType {
  def elementType: PType

  def asPContainer: PContainer = this match {
    case _: PStream => PArray(this.elementType, this.required)
    case x: PContainer => x
  }
}
