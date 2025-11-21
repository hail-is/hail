package is.hail.utils.compat.mutable

trait GrowableCompat[-A] extends Growable[A] {
  override def +=(elem: A): this.type = addOne(elem)
  protected def addOne(elem: A): this.type
  def knownSize: Int = -1
}
