package is.hail.utils

import java.util
import java.util.Map.Entry

class Cache[K, V](capacity: Int) {
  val m = new util.LinkedHashMap[K, V](capacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[K, V]): Boolean = size() > capacity
  }

  def get(k: K): Option[V] = Option(m.get(k))

  def +=(p: (K, V)): Unit = m.put(p._1, p._2)

  def size: Int = m.size()
}
