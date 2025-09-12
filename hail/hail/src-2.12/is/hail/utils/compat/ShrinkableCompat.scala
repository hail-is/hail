package is.hail.utils.compat

import scala.collection.generic.Shrinkable

trait ShrinkableCompat[-A] extends Shrinkable[A] {
  override def -=(elem: A): this.type = subtractOne(elem)
  protected def subtractOne(elem: A): this.type
}
