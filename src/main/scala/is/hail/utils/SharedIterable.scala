package is.hail.utils

import scala.annotation.tailrec
import scala.collection.generic.CanBuildFrom
import scala.collection.{Iterable, mutable}
import scala.reflect.ClassTag

object SharedIterable {

  def empty[T]: SharedIterable[T] = new SharedIterable[T] {
    override def iterator: SharedIterator[Nothing] = SharedIterator.empty
  }

  implicit def canBuildFrom[T](implicit tct: ClassTag[T]): CanBuildFrom[Nothing, T, SharedIterable[T]] =
    new CanBuildFrom[Nothing, T, SharedIterable[T]]() {
      def apply(x: Nothing): mutable.Builder[T, SharedIterable[T]] = new SharedIterableBuilder[T]()

      def apply(): mutable.Builder[T, SharedIterable[T]] = new SharedIterableBuilder[T]()
    }
}

abstract class SharedIterable[+T] {
  self =>

  def iterator: SharedIterator[T]

  // FIXME: temporary to allow building, unsafe once we switch to MutableGenotypeStreamIterator
  def toIterable: Iterable[T] = new Iterable[T] {
    override def iterator = new Iterator[T] {
      private val it = self.iterator

      override def hasNext: Boolean = it.hasNext

      override def next(): T = it.next()
    }
  }

  def lazyFilterWith[T2](i2: Iterable[T2], p: (T, T2) => Boolean): SharedIterable[T] =
    new SharedIterable[T] with Serializable {
      def iterator: SharedIterator[T] = new SharedIterator[T] {
        val it: SharedIterator[T] = self.iterator
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

  def ++[U >: T](that: => SharedIterable[U]): SharedIterable[U] = new SharedIterable[U] {
    def iterator: SharedIterator[U] = self.iterator ++ that.iterator // FIXME: should be () => ?
  }

  def foldLeft[U](z: U)(op: (U, T) => U): U = self.iterator.foldLeft(z)(op)

  def foreach[U](f: T => U): Unit = self.iterator.foreach(f)

  def lazyMap[S](f: (T) => S): SharedIterable[S] = new SharedIterable[S] with Serializable {
    def iterator: SharedIterator[S] = new SharedIterator[S] {
      val it: SharedIterator[T] = self.iterator

      def hasNext: Boolean = it.hasNext

      def next(): S = f(it.next())
    }
  }

  def zip[U](that: SharedIterable[U]): SharedIterable[(T, U)] = new SharedIterable[(T,U)] {
    def iterator: SharedIterator[(T, U)] = self.iterator.zip(that.iterator)
  }

  def forall(p: T => Boolean): Boolean = self.iterator.forall(p)
}


object SharedIterator {
  def empty[T]: SharedIterator[T] = new SharedIterator[T] {
    def hasNext: Boolean = false
    def next(): Nothing = throw new NoSuchElementException("next on empty iterator")
  }
}

abstract class SharedIterator[+T] {
  self =>

  def next(): T

  def hasNext: Boolean

  def foldLeft[U](z: U)(op: (U, T) => U): U = {
    var result = z
    self.foreach(x => result = op(result, x))
    result
  }

  def foreach[U](f: T => U) { while (hasNext) f(next()) }

  def map[U](f: T => U): SharedIterator[U] = new SharedIterator[U] {
    def hasNext: Boolean = self.hasNext
    def next(): U = f(self.next())
  }

  def zip[U](that: SharedIterator[U]): SharedIterator[(T, U)] = new SharedIterator[(T, U)] {
    def hasNext: Boolean = self.hasNext && that.hasNext
    def next(): (T, U) = (self.next(), that.next())
  }

  def zipWithIndex: SharedIterator[(T, Int)] = new SharedIterator[(T, Int)] {
    var idx = 0

    def hasNext: Boolean = self.hasNext

    def next(): (T, Int)  = {
      val ret = (self.next(), idx)
      idx += 1
      ret
    }
  }

  def ++[U >: T](that: => SharedIterator[U]): SharedIterator[U] =
    new ConcatSharedIterator(self, Vector(() => that))

  def forall(p: T => Boolean): Boolean = {
    var res = true
    while (res && hasNext) res = p(next())
    res
  }
}

final class ConcatSharedIterator[+T](private[this] var current: SharedIterator[T], initial: Vector[() => SharedIterator[T]]) extends SharedIterator[T] {
  private[this] var queue: Vector[() => SharedIterator[T]] = initial
  private[this] var currentHasNextChecked = false

  @tailrec
  private[this] def advance(): Boolean = {
    if (queue.isEmpty) {
      current = null
      false
    }
    else {
      current = queue.head()
      queue = queue.tail
      if (current.hasNext) {
        currentHasNextChecked = true
        true
      } else advance()
    }
  }
  def hasNext: Boolean =
    if (currentHasNextChecked) true
    else if (current eq null) false
    else if (current.hasNext) {
      currentHasNextChecked = true
      true
    } else advance()

  def next(): T =
    if (hasNext) {
      currentHasNextChecked = false
      current.next()
    } else Iterator.empty.next()

  override def ++[U >: T](that: => SharedIterator[U]): SharedIterator[U] =
    new ConcatSharedIterator(current, queue :+ (() => that))
}

class SharedIterableBuilder[T](implicit tct: ClassTag[T]) extends mutable.Builder[T, SharedIterable[T]] {

  val b: ArrayBuilder[T] = new ArrayBuilder[T]()

  override def +=(g: T): SharedIterableBuilder.this.type = {
    b += g
    this
  }

  def ++=(i: SharedIterable[T]): SharedIterableBuilder[T] = {
    i.foreach(this += _)
    this
  }

  override def clear() { b.clear() }

  override def result(): SharedIterable[T] = b.result().toSharedIterable
}