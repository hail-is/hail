package is.hail.types

import java.util.function._

final case class Box[T](
  b: ThreadLocal[Option[T]] = ThreadLocal.withInitial(
    new Supplier[Option[T]] { def get = None }
  ),
  matchCond: (T, T) => Boolean = { (a: T, b: T) => a == b },
) {
  def unify(t: T): Boolean = b.get match {
    case Some(bt) => matchCond(t, bt)
    case None =>
      b.set(Some(t))
      true
  }

  def clear() {
    b.set(None)
  }

  def get: T = b.get.get

  def isEmpty: Boolean = b.get.isEmpty
  def isDefined: Boolean = b.get.isDefined
}
