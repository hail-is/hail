package is.hail.utils.richUtils

import java.io.Serializable

import is.hail.utils._

import scala.collection.{TraversableOnce, mutable}

class RichIterable[T](val i: Iterable[T]) extends Serializable {
  def lazyMap[S](f: (T) => S): Iterable[S] = new Iterable[S] with Serializable {
    def iterator: Iterator[S] = new Iterator[S] {
      val it: Iterator[T] = i.iterator

      def hasNext: Boolean = it.hasNext

      def next(): S = f(it.next())
    }
  }

  def foreachBetween(f: (T) => Unit)(g: => Unit) {
    i.iterator.foreachBetween(f)(g)
  }

  def lazyMapWith[T2, S](i2: Iterable[T2], f: (T, T2) => S): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator

        def hasNext: Boolean = it.hasNext && it2.hasNext

        def next(): S = f(it.next(), it2.next())
      }
    }

  def lazyMapWith2[T2, T3, S](i2: Iterable[T2], i3: Iterable[T3], f: (T, T2, T3) => S): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator
        val it3: Iterator[T3] = i3.iterator

        def hasNext: Boolean = it.hasNext && it2.hasNext && it3.hasNext

        def next(): S = f(it.next(), it2.next(), it3.next())
      }
    }

  def lazyFilterWith[T2](i2: Iterable[T2], p: (T, T2) => Boolean): Iterable[T] =
    new Iterable[T] with Serializable {
      def iterator: Iterator[T] = new Iterator[T] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator

        var pending: Boolean = false
        var pendingNext: T = _

        def setNext() {
          while (!pending && it.hasNext && it2.hasNext) {
            val n = it.next()
            val n2 = it2.next()
            if (p(n, n2)) {
              pending = true
              pendingNext = n
            }
          }
        }

        def hasNext: Boolean = {
          setNext()
          pending
        }

        def next(): T = {
          if (!pending)
            setNext()
          pending = false
          pendingNext
        }
      }
    }

  def lazyFlatMap[S](f: (T) => TraversableOnce[S]): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        var current: Iterator[S] = Iterator.empty

        def hasNext: Boolean =
          if (current.hasNext)
            true
          else {
            if (it.hasNext) {
              current = f(it.next()).toIterator
              hasNext
            } else
              false
          }

        def next(): S = current.next()
      }
    }

  def lazyFlatMapWith[S, T2](i2: Iterable[T2], f: (T, T2) => TraversableOnce[S]): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator
        var current: Iterator[S] = Iterator.empty

        def hasNext: Boolean =
          if (current.hasNext)
            true
          else {
            if (it.hasNext && it2.hasNext) {
              current = f(it.next(), it2.next()).toIterator
              hasNext
            } else
              false
          }

        def next(): S = current.next()
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
    def truncate: String = if (i.size > toTake)
      i.take(toTake).mkString(delim) + delim + "..."
    else
      i.mkString(delim)

    def strings: (String, String) = (truncate, i.mkString(delim))
  }

  def counter(): Map[T, Int] = {
    val m = new mutable.HashMap[T, Int]()
    i.foreach { elem => m.updateValue(elem, 0, _ + 1) }

    m.toMap
  }
}
