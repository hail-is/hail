package is.hail.expr.types

final case class Box[T](var b: Option[T] = None, matchCond: (T, T) => Boolean = { (a: T, b: T) => a == b }) {
  def unify(t: T): Boolean = b match {
    case Some(bt) => matchCond(t, bt)
    case None =>
      b = Some(t)
      true
  }

  def clear() {
    b = None
  }

  def get: T = b.get

  def isEmpty: Boolean = b.isEmpty
  def isDefined: Boolean = b.isDefined
}
