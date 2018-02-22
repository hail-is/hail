package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test
import scala.collection.generic.Growable
import scala.collection.mutable.ArrayBuffer

class EphemeralIteratorSuite extends SparkSuite {
  class Box[A] extends AnyRef {
    var value: A = _

    def canEqual(a: Any): Boolean = a.isInstanceOf[Box[A]]
    override def equals(that: Any): Boolean =
      that match {
        case that: Box[A] => value == that.value
        case _ => false
      }

  }

  object Box {
    def apply[A](): Box[A] = new Box
    def apply[A](a: A): Box[A] = {
      val box = Box[A]()
      box.value = a
      box
    }
  }

  def boxEquiv[A]: EquivalenceClassView[Box[A]] = new EquivalenceClassView[Box[A]] {
    var value: A = _
    def setNonEmptyEquivClass(a: Box[A]) {
      value = a.value
    }
    def inNonEmptyEquivClass(a: Box[A]): Boolean = {
      a.value == value
    }
  }

  def boxBuffer[A]: Growable[Box[A]] with Iterable[Box[A]] =
    new Growable[Box[A]] with Iterable[Box[A]] {
      val buf = ArrayBuffer[A]()
      val box = Box[A]()
      def clear() { buf.clear(); i = 0 }
      def +=(x: Box[A]) = {
        buf += x.value
        this
      }
      var i: Int = 0
      def iterator: Iterator[Box[A]] = new Iterator[Box[A]] {
        def hasNext = i < buf.size
        def next() = {
          box.value = buf(i)
          i += 1
          box
        }
      }
    }

  def boxIntOrd: (Box[Int], Box[Int]) => Int =
    (l, r) => l.value - r.value

  def makeTestIterator[A](elems: A*): StagingIterator[Box[A]] = {
    val it = elems.iterator
    val sm = new StateMachine[Box[A]] {
      val curValue: Box[A] = Box()
      var isActive = true
      def advance() {
        if (it.hasNext)
          curValue.value = it.next()
        else
          isActive = false
      }
    }
    sm.advance()
    StagingIterator(sm)
  }

  @Test def ephemeralIteratorStartsWithRightValue() {
    val it: EphemeralIterator[Box[Int]] =
      makeTestIterator(1, 2, 3, 4, 5)
    assert(it.value.value == 1)
  }

  @Test def makeTestIteratorWorks() {
    val testIt = makeTestIterator(1, 2, 3, 4, 5)
    val it = Iterator.range(1, 6).map(Box[Int](_))
    assert(testIt sameElements it)

    val emptyTest = makeTestIterator()
    val emptyIt = Iterator.empty
    assert(emptyTest sameElements emptyIt)
  }

  @Test def toStaircaseWorks() {
    val testIt = makeTestIterator(1, 1, 2, 3, 3, 3)
    val it = makeTestIterator(
      makeTestIterator(1, 1),
      makeTestIterator(2),
      makeTestIterator(3, 3, 3))
      .map(_.value)
    assert(testIt.staircased(boxEquiv[Int]).compareUsing[StagingIterator[Box[Int]]](it, _.sameElements(_)))
  }

  @Test def orderedZipJoinWorks() {
    val left = makeTestIterator(1, 2, 4)
    val right = makeTestIterator(2, 3, 4)
    val zipped = left.orderedZipJoin(
      right,
      Box(0),
      Box(0),
      boxIntOrd)

    val muple = Muple(Box(0), Box(0))
    val shouldBe = Iterator((1, 0), (2, 2), (0, 3), (4, 4))
      .map{case (x, y) => { muple._1.value = x; muple._2.value = y; muple }}

    assert(zipped.sameElements(shouldBe))
  }

  @Test def innerJoinDistinctWorks() {
    val left = makeTestIterator(1, 2, 2, 4)
    val right = makeTestIterator(2, 4, 4, 5)
    val joined = left.innerJoinDistinct(
      right,
      boxEquiv[Int],
      boxEquiv[Int],
      Box(0),
      Box(0),
      boxIntOrd
    )

    val muple = Muple(Box(0), Box(0))
    val shouldBe = Iterator((2, 2), (2, 2), (4, 4))
    .map { case (x, y) => { muple._1.value = x; muple._2.value = y; muple } }
  }

  @Test def leftJoinDistinctWorks() {
    val left = makeTestIterator(1, 2, 2, 4)
    val right = makeTestIterator(2, 4, 4, 5)
    val joined = left.innerJoinDistinct(
      right,
      boxEquiv[Int],
      boxEquiv[Int],
      Box(0),
      Box(0),
      boxIntOrd
    )

    val muple = Muple(Box(0), Box(0))
    val shouldBe = Iterator((1, 0), (2, 2), (2, 2), (4, 4))
      .map { case (x, y) => { muple._1.value = x; muple._2.value = y; muple } }
  }

  @Test def innerJoinWorks() {
    val left = makeTestIterator(1, 2, 2, 4, 5, 5)
    val right = makeTestIterator(2, 2, 4, 4, 5, 6)
    val joined = left.innerJoin(
      right,
      boxEquiv[Int],
      boxEquiv[Int],
      Box(0),
      Box(0),
      boxBuffer[Int],
      boxIntOrd
    )

    val muple = Muple(Box(0), Box(0))
    val shouldBe = Iterator((2, 2), (2, 2), (2, 2), (2, 2), (4, 4), (4, 4), (5, 5), (5, 5))
      .map { case (x, y) => { muple._1.value = x; muple._2.value = y; muple } }
  }

  @Test def leftJoinWorks() {
    val left = makeTestIterator(1, 2, 2, 4, 5, 5)
    val right = makeTestIterator(2, 2, 4, 4, 5, 6)
    val joined = left.leftJoin(
      right,
      boxEquiv[Int],
      boxEquiv[Int],
      Box(0),
      Box(0),
      boxBuffer[Int],
      boxIntOrd
    )

    val muple = Muple(Box(0), Box(0))
    val shouldBe = Iterator((1, 0), (2, 2), (2, 2), (2, 2), (2, 2), (4, 4), (4, 4), (5, 5), (5, 5))
      .map { case (x, y) => { muple._1.value = x; muple._2.value = y; muple } }
  }
}
