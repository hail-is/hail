package is.hail.utils.richUtils

class RichOption[T](val o: Option[T]) extends AnyVal {
  def contains(v: T): Boolean = o.isDefined && o.get == v

  override def toString: String = o.toString

  def liftedZip[U](other: Option[U]): Option[(T, U)] = o.flatMap { val1 =>
    other.map(val2 => (val1, val2))
  }
}
