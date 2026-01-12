package is.hail.collection

import is.hail.collection.compat.mutable.{GrowableCompat, ShrinkableCompat}

import scala.collection.mutable

object ImmutableMap {

  private[this] object EmptyInstance
      extends mutable.AbstractMap[Any, Any] with GrowableCompat[(Any, Any)]
      with ShrinkableCompat[Any] {
    override def addOne(kv: (Any, Any)): EmptyInstance.this.type = this
    override def subtractOne(key: Any): EmptyInstance.this.type = this
    override def get(key: Any): Option[Any] = None
    override def iterator: Iterator[(Any, Any)] = Iterator.empty
    override def getOrElseUpdate(key: Any, op: => Any): Any = op
  }

  def empty[K, V]: mutable.Map[K, V] =
    EmptyInstance.asInstanceOf[mutable.Map[K, V]]
}
