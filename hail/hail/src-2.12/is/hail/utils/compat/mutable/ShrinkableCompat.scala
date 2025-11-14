package is.hail.utils.compat.mutable

trait ShrinkableCompat[-A] extends Shrinkable[A] {
  override def -=(elem: A): this.type = subtractOne(elem)
  protected def subtractOne(elem: A): this.type
}
