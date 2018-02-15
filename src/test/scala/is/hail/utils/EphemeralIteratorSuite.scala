package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

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
  def makeTestIterator[A](elems: A*): StagingIterator[Box[A]] = {
    val it = elems.iterator
    new StagingIterator[Box[A]] {
      val head: Box[A] = Box()
      var isValid = true
      def advanceHead() {
        if (it.hasNext)
          head.value = it.next()
        else
          isValid = false
      }
      advanceHead()
    }
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

}
