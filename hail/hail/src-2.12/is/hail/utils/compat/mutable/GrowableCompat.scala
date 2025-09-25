package is.hail.utils.compat.mutable

import scala.collection.generic

trait Growable[-A] extends generic.Growable[A] {
  override def +=(elem: A): this.type = addOne(elem)
  protected def addOne(elem: A): this.type
  def knownSize: Int = -1
}
