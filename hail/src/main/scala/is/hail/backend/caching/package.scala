package is.hail.backend

import scala.collection.mutable

package object caching {
  private[this] object NoCachingInstance extends mutable.AbstractMap[Any, Any] {
    override def +=(kv: (Any, Any)): NoCachingInstance.this.type = this
    override def -=(key: Any): NoCachingInstance.this.type = this
    override def get(key: Any): Option[Any] = None
    override def iterator: Iterator[(Any, Any)] = Iterator.empty
    override def getOrElseUpdate(key: Any, op: => Any): Any = op
  }

  def NoCaching[K, V]: mutable.Map[K, V] =
    NoCachingInstance.asInstanceOf[mutable.Map[K, V]]
}
