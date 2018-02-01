package is.hail.utils

trait MutableEquiv[T] {
  def setEquivClass(t: T): Unit
  def inEquivClass(t: T): Boolean
}
