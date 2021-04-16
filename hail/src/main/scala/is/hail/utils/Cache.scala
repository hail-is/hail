package is.hail.utils

import java.util
import java.util.Map.Entry

class Cache[K, V](capacity: Int) {
  private[this] val m = new util.LinkedHashMap[K, V](capacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[K, V]): Boolean = size() > capacity
  }

  def get(k: K): Option[V] = synchronized { Option(m.get(k)) }

  def +=(p: (K, V)): Unit = synchronized { m.put(p._1, p._2) }

  def size: Int = synchronized { m.size() }
}
