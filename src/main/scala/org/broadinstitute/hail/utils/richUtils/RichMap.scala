package org.broadinstitute.hail.utils.richUtils

class RichMap[K, V](val m: Map[K, V]) extends AnyVal {
  def mapValuesWithKeys[T](f: (K, V) => T): Map[K, T] = m map { case (k, v) => (k, f(k, v)) }

  def force = m.map(identity) // needed to make serializable: https://issues.scala-lang.org/browse/SI-7005
}
