package is.hail.collection.implicits

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.utils._

import scala.collection.{mutable, AbstractIterable}
import scala.reflect.ClassTag

object RichIterable {
  def single[A](a: A): Iterable[A] = new AbstractIterable[A] {
    override def iterator = Iterator.single(a)
    override def head = a
    override def headOption = Some(a)
    override def last = a
    override def lastOption = Some(a)
    override def take(n: Int) = if (n > 0) this else Iterable.empty
    override def takeRight(n: Int) = if (n > 0) this else Iterable.empty
    override def drop(n: Int) = if (n > 0) Iterable.empty else this
    override def dropRight(n: Int) = if (n > 0) Iterable.empty else this
    override def tail = Iterable.empty
    override def init = Iterable.empty
  }
}

class RichIterable[T](val i: Iterable[T]) extends Serializable {
  def foreachBetween(f: (T) => Unit)(g: => Unit): Unit =
    i.iterator.foreachBetween(f)(g)

  def intersperse[S >: T](sep: S): Iterable[S] = new Iterable[S] {
    override def iterator = i.iterator.intersperse(sep)
  }

  def intersperse[S >: T](start: S, sep: S, end: S): Iterable[S] = new Iterable[S] {
    override def iterator = i.iterator.intersperse(start, sep, end)
  }

  def +:[S >: T](elt: S): Iterable[S] = new Iterable[S] {
    override def iterator = Iterator.single(elt) ++ i
  }

  def lazyMapWith[T2, S](i2: Iterable[T2], f: (T, T2) => S): Iterable[S] =
    new Iterable[S] with Serializable {
      override def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator

        override def hasNext: Boolean = it.hasNext && it2.hasNext

        override def next(): S = f(it.next(), it2.next())
      }
    }

  def lazyMapWith2[T2, T3, S](i2: Iterable[T2], i3: Iterable[T3], f: (T, T2, T3) => S)
    : Iterable[S] =
    new Iterable[S] with Serializable {
      override def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator
        val it3: Iterator[T3] = i3.iterator

        override def hasNext: Boolean = it.hasNext && it2.hasNext && it3.hasNext

        override def next(): S = f(it.next(), it2.next(), it3.next())
      }
    }

  def areDistinct(): Boolean = {
    val seen = mutable.HashSet[T]()
    for (x <- i)
      if (seen(x))
        return false
      else
        seen += x
    true
  }

  def duplicates(): Set[T] = {
    val dups = mutable.HashSet[T]()
    val seen = mutable.HashSet[T]()
    for (x <- i)
      if (seen(x))
        dups += x
      else
        seen += x
    dups.toSet
  }

  def truncatable(delim: String = ", ", toTake: Int = 10): Truncatable = new Truncatable {
    override def truncate: String = if (i.size > toTake)
      i.take(toTake).mkString(delim) + delim + "..."
    else
      i.mkString(delim)

    override def strings: (String, String) = (truncate, i.mkString(delim))
  }

  def counter(): Map[T, Int] = {
    val m = new mutable.HashMap[T, Int]()
    i.foreach(elem => m.updateValue(elem, 0, _ + 1))
    m.toMap
  }

  def toFastSeq(implicit tct: ClassTag[T]): ArraySeq[T] =
    i.to(ArraySeq)
}
