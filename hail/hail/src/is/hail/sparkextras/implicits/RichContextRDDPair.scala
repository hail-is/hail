package is.hail.sparkextras.implicits

import is.hail.sparkextras.ContextRDD

import scala.reflect.ClassTag

class RichContextRDDPair[K, V](val crdd: ContextRDD[(K, V)]) extends AnyVal {
  def values(implicit VT: ClassTag[V]): ContextRDD[V] =
    crdd.map(_._2)
}
