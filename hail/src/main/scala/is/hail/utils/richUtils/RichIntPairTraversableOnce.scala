package is.hail.utils.richUtils

import scala.collection.TraversableOnce
import scala.reflect.ClassTag

class RichIntPairTraversableOnce[V](val t: TraversableOnce[(Int, V)]) extends AnyVal {
  def reduceByKeyToArray(n: Int, zero: => V)(f: (V, V) => V)(implicit vct: ClassTag[V])
    : Array[V] = {
    val a = Array.fill[V](n)(zero)
    t.foreach { case (k, v) =>
      a(k) = f(a(k), v)
    }
    a
  }
}
