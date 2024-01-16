package is.hail.utils.richUtils

import scala.collection.mutable

class RichMutableMap[K, V](val m: mutable.Map[K, V]) extends AnyVal {
  def updateValue(k: K, default: => V, f: (V) => V): Unit =
    m += ((k, f(m.getOrElse(k, default))))
}
