package is.hail.utils.richUtils

import scala.collection.compat._
import scala.reflect.ClassTag

class RichIntPairIterableOnce[V](val t: IterableOnce[(Int, V)]) extends AnyVal {
  def reduceByKeyToArray(n: Int, zero: => V)(f: (V, V) => V)(implicit vct: ClassTag[V])
    : Array[V] = {
    val a = Array.fill[V](n)(zero)
    t.foreach { case (k, v) =>
      a(k) = f(a(k), v)
    }
    a
  }
}
