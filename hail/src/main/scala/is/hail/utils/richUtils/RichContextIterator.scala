package is.hail.utils.richUtils

import is.hail.utils.WithContext

class RichContextIterator[T](val i: Iterator[WithContext[T]]) {
  def mapLines[U](f: T => U): Iterator[U] = i.map(_.map(f).value)

  def foreachLine(f: T => Unit): Unit =
    i.foreach(_.foreach(f))
}
