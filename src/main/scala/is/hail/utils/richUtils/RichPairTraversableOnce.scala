package is.hail.utils.richUtils

import scala.collection.{TraversableOnce, mutable}

class RichPairTraversableOnce[K, V](val t: TraversableOnce[(K, V)]) extends AnyVal {
  def reduceByKey(f: (V, V) => V): scala.collection.Map[K, V] = {
    val m = mutable.Map.empty[K, V]
    t.foreach { case (k, v) =>
      m.get(k) match {
        case Some(v2) => m += k -> f(v, v2)
        case None => m += k -> v
      }
    }
    m
  }
}
