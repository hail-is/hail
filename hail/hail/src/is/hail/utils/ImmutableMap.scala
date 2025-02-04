package is.hail.utils

import scala.collection.mutable

object ImmutableMap {

  private[this] object EmptyInstance extends mutable.AbstractMap[Any, Any] {
    override def +=(kv: (Any, Any)): EmptyInstance.this.type = this
    override def -=(key: Any): EmptyInstance.this.type = this
    override def get(key: Any): Option[Any] = None
    override def iterator: Iterator[(Any, Any)] = Iterator.empty
    override def getOrElseUpdate(key: Any, op: => Any): Any = op
  }

  def empty[K, V]: mutable.Map[K, V] =
    EmptyInstance.asInstanceOf[mutable.Map[K, V]]
}
