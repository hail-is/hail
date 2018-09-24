package is.hail.utils

import scala.reflect.ClassTag

abstract class HailIterator[@specialized T] {
  def next(): T

  def hasNext: Boolean

  def toArray(implicit tct: ClassTag[T]): Array[T] = {
    val b = new ArrayBuilder[T]()
    while (hasNext)
      b += next()
    b.result()
  }

  def countNonNegative()(implicit ev: Numeric[T]): Int = {
    import ev._
    var count = 0
    while (hasNext) {
      if (next() >= ev.zero) count += 1
    }
    count
  }
}
