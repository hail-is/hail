package is.hail.utils

trait EquivalenceClassView[A] {
  def setNonEmptyEquivClass(a: A): Unit
  def inNonEmptyEquivClass(a: A): Boolean

  def setEquivClass(a: A): Unit = {
    if (a == null)
      isEmpty = true
    else {
      isEmpty = false
      setNonEmptyEquivClass(a)
    }
  }
  def inEquivClass(a: A): Boolean = !isEmpty && inNonEmptyEquivClass(a)
  def setEmpty(): Unit = isEmpty = true
  private var isEmpty = true
}

object EquivalenceClassView {
  def fromEquiv[A](implicit equiv: Equiv[A]): EquivalenceClassView[A] = new EquivalenceClassView[A] {
    private var a: A = _
    def setNonEmptyEquivClass(a: A) = this.a = a
    def inNonEmptyEquivClass(a: A) = equiv.equiv(this.a, a)
  }
}
