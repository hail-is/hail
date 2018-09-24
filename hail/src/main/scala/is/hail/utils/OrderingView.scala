package is.hail.utils

trait OrderingView[A] {
  protected def setFiniteValue(a: A): Unit
  protected def compareFinite(a: A): Int

  def setValue(a: A) {
    isInfinite = 0
    setFiniteValue(a)
  }
  def compare(a: A): Int =
    if (isInfinite != 0) isInfinite
    else compareFinite(a)

  def isEquivalent(a: A): Boolean = compare(a) == 0
  def setBottom() { isInfinite = -1 }
  def setTop() { isInfinite = 1 }
  private var isInfinite: Int = -1
}

object OrderingView {
  def fromOrdering[A](implicit ord: Ordering[A]): OrderingView[A] =
    new OrderingView[A] {
      private var a: A = _
      def setFiniteValue(a: A) { this.a = a }
      def compareFinite(a: A) = ord.compare(this.a, a)
    }
}
