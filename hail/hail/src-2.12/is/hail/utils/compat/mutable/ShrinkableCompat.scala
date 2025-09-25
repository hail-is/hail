package is.hail.utils.compat.mutable

import scala.collection.generic

trait Shrinkable[-A] extends generic.Shrinkable[A] {
  override def -=(elem: A): this.type = subtractOne(elem)
  protected def subtractOne(elem: A): this.type
}
