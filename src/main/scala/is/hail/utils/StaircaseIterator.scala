package is.hail.utils

import scala.collection.{BufferedIterator, Iterator}
import scala.collection.mutable.ArrayBuffer

import is.hail.annotations.{JoinedRegionValue, OrderedRVIterator, Region, RegionValue, RegionValueBuilder, UnsafeOrdering}
import is.hail.rvd.OrderedRVType

class StaircaseIterator[A](it: BufferedIterator[A], equiv: EquivalenceClassView[A])
  extends Iterator[BufferedIterator[A]] {

  equiv.setEmpty()

  private object stepIterator extends BufferedIterator[A] {
    def hasNext: Boolean = it.hasNext && equiv.inEquivClass(it.head)
    def head: A = { assert(hasNext); it.head }
    def next(): A = { assert(hasNext); it.next() }
    def exhaust(): Unit = while (hasNext) next()
  }

  def hasNext: Boolean = {
    stepIterator.exhaust()
    it.hasNext
  }

  def next(): BufferedIterator[A] = {
    val b = hasNext
    assert(b)
    equiv.setEquivClass(it.head)
    stepIterator
  }
}

class TestIterator[A](seq: Seq[A])
    extends BufferedIterator[A] {
  var i = 0
  def hasNext: Boolean = i < seq.length
  def head: A = seq(i)
  def next(): A = {
    i += 1
    seq(i - 1)
  }
}
