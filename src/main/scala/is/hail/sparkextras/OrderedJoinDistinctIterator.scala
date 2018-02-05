package is.hail.sparkextras

import scala.collection.{BufferedIterator, Iterator}
import scala.collection.mutable.ArrayBuffer

import is.hail.annotations.{JoinedRegionValue, OrderedRVIterator, Region, RegionValue, RegionValueBuilder, UnsafeOrdering}
import is.hail.rvd.OrderedRVDType
import is.hail.utils.{Muple, MutableEquiv}

abstract class OrderedJoinIterator(
    leftTyp: OrderedRVDType,
    rightTyp: OrderedRVDType,
    leftIt: Iterator[RegionValue],
    rightIt: Iterator[RegionValue])
  extends Iterator[JoinedRegionValue] {

  private val jrv = JoinedRegionValue()
  private val rCache = Region()
  private val rCacheIdx = ArrayBuffer[Long]()
  private val rvb = new RegionValueBuilder(rCache)

  private var lrv: RegionValue = _
  private var rrv: RegionValue = if (rightIt.hasNext) rightIt.next() else null

  private var jrvPresent = false
  private def rCachePresent = false
  private var rCacheCur = -1
  private val lrKOrd =
    OrderedRVDType.selectUnsafeOrdering(
      leftTyp.rowType, leftTyp.kRowFieldIdx,
      rightTyp.rowType, rightTyp.kRowFieldIdx)

  def lrCompare: Int = {
    assert(lrv != null && rrv != null)
    lrKOrd.compare(lrv, rrv)
  }

  def isPresent: Boolean = jrvPresent

  protected def advance() {
    if (rCachePresent) {
      assert(lrCompare == 0)
      if (rCacheCur < rCacheIdx.size) {
        rCacheCur += 1
        rrv.set(rCache, rCacheIdx(rCacheCur))
      }
      else {
        advanceLeft1()
        if (lrCompare == 0) {
          rCacheCur = 0
          rrv.set(rCache, rCacheIdx(rCacheCur))
        }
        else {

        }
      }
    }
    else if (lrCompare < 0)
      advanceLeft1()
    else if (lrCompare > 0) {
      assert(!rCachePresent)
      advanceRight1()
    }
    else {  // lrCompare == 0
      cacheRight()
      advanceRight1()
      if (lrCompare < 0) {
        advance
      }
    }
  }

  def cacheRight() {
    rvb.start(rightTyp.rowType)
    rvb.addRegionValue(rightTyp.rowType, rrv)
    rCacheIdx += rvb.end()
  }

  def advanceLeft1() {
    lrv = if (leftIt.hasNext) leftIt.next() else null
  }

  def advanceLeft() {
    assert(rrv != null)
    while (lrv != null && lrKOrd.compare(lrv, rrv) < 0) {
      advanceLeft1()
    }
  }

  def hasLeft: Boolean = lrv != null

  def advanceRight1() {
    rrv = if (rightIt.hasNext) rightIt.next() else null
  }

  def advanceRight() {
    assert(lrv != null)
    while (rrv != null && lrKOrd.compare(lrv, rrv) > 0) {
      advanceRight1()
    }
  }

  def hasRight: Boolean = rrv != null

  def setJRV() {
    jrv.set(lrv, rrv)
    jrvPresent = true
  }

  def setJRVRightNull() {
    jrv.set(lrv, null)
    jrvPresent = true
  }

  def setJRVLeftNull() {
    jrv.set(null, rrv)
    jrvPresent = true
  }

  def hasNext: Boolean

  def next(): JoinedRegionValue = {
    if (!hasNext)
      throw new NoSuchElementException("next on empty iterator")
    jrvPresent = false
    jrv
  }
}


abstract class OrderedJoinDistinctIterator(
    leftTyp: OrderedRVDType,
    rightTyp: OrderedRVDType,
    leftIt: Iterator[RegionValue],
    rightIt: Iterator[RegionValue])
  extends Iterator[JoinedRegionValue] {

  private val jrv = JoinedRegionValue()
  private var lrv: RegionValue = _
  private var rrv: RegionValue = if (rightIt.hasNext) rightIt.next() else null

  private var jrvPresent = false
  private val lrKOrd = OrderedRVDType.selectUnsafeOrdering(leftTyp.rowType, leftTyp.kRowFieldIdx, rightTyp.rowType, rightTyp.kRowFieldIdx)

  def lrCompare(): Int = {
    assert(lrv != null && rrv != null)
    lrKOrd.compare(lrv, rrv)
  }

  def isPresent: Boolean = jrvPresent

  def advanceLeft1() {
    lrv = if (leftIt.hasNext) leftIt.next() else null
  }

  def advanceLeft() {
    assert(rrv != null)
    while (lrv != null && lrKOrd.compare(lrv, rrv) < 0) {
      advanceLeft1()
    }
  }

  def hasLeft: Boolean = lrv != null

  def advanceRight1() {
    rrv = if (rightIt.hasNext) rightIt.next() else null
  }

  def advanceRight() {
    assert(lrv != null)
    while (rrv != null && lrKOrd.compare(lrv, rrv) > 0) {
      advanceRight1()
    }
  }

  def hasRight: Boolean = rrv != null

  def setJRV() {
    jrv.set(lrv, rrv)
    jrvPresent = true
  }

  def setJRVRightNull() {
    jrv.set(lrv, null)
    jrvPresent = true
  }

  def setJRVLeftNull() {
    jrv.set(null, rrv)
    jrvPresent = true
  }

  def hasNext: Boolean

  def next(): JoinedRegionValue = {
    if (!hasNext)
      throw new NoSuchElementException("next on empty iterator")
    jrvPresent = false
    jrv
  }
}

// class OrderedInnerJoinDistinctIterator(
//     leftTyp: OrderedRVDType,
//     rightTyp: OrderedRVDType,
//     leftIt: Iterator[RegionValue],
//     rightIt: Iterator[RegionValue])
//   extends Iterator[JoinedRegionValue] {

//   val left = OrderedRVIterator(leftTyp, leftIt.buffered)
//   val right = OrderedRVIterator(rightTyp, rightIt.buffered)
//   val jrv = JoinedRegionValue()

//   val it = for (Muple(l, r) <- left.cogroup(right) if (l.hasNext && r.hasNext))
//   yield {
//     jrv.set(l.head, r.head)
//     jrv
//   }

//   def hasNext: Boolean = it.hasNext
//   def next(): JoinedRegionValue = it.next()
// }

class OrderedInnerJoinDistinctIterator(
    leftTyp: OrderedRVDType,
    rightTyp: OrderedRVDType,
    leftIt: Iterator[RegionValue],
    rightIt: Iterator[RegionValue])
  extends OrderedJoinDistinctIterator(leftTyp, rightTyp, leftIt, rightIt) {

  def hasNext: Boolean = {
    if (!isPresent) {
      advanceLeft1()
      while (!isPresent && hasLeft && hasRight) {
        val c = lrCompare()
        if (c == 0)
          setJRV()
        else if (c > 0)
          advanceRight()
        else
          advanceLeft()
      }
    }

    isPresent
  }
}

class OrderedLeftJoinDistinctIterator(
    leftTyp: OrderedRVDType,
    rightTyp: OrderedRVDType,
    leftIt: Iterator[RegionValue],
    rightIt: Iterator[RegionValue])
  extends Iterator[JoinedRegionValue] {

  val left = OrderedRVIterator(leftTyp, leftIt.buffered)
  val right = OrderedRVIterator(rightTyp, rightIt.buffered)
  val jrv = JoinedRegionValue()

  val it = for {
    Muple(l, r) <- left.cogroup(right)
    lrv <- l
  } yield {
    if (r.hasNext)
      jrv.set(lrv, r.head)
    else
      jrv.set(lrv, null)
    jrv
  }

  def hasNext: Boolean = it.hasNext
  def next(): JoinedRegionValue = it.next()
}

// class OrderedLeftJoinDistinctIterator(
//     leftTyp: OrderedRVType,
//     rightTyp: OrderedRVType,
//     leftIt: Iterator[RegionValue],
//     rightIt: Iterator[RegionValue])
//   extends OrderedJoinDistinctIterator(leftTyp, rightTyp, leftIt, rightIt) {

//   def hasNext: Boolean = {
//     if (!isPresent) {
//       advanceLeft1()
//       while (!isPresent && hasLeft) {
//         if (!hasRight || lrCompare() < 0)
//           setJRVRightNull()
//         else if (lrCompare() == 0)
//           setJRV()
//         else
//           advanceRight()
//       }
//     }

//     isPresent
//   }
// }

class StaircaseIterator[A](it: BufferedIterator[A], equiv: MutableEquiv[A])
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
  def hasNext: Boolean = i < seq.length - 1
  def head: A = seq(i)
  def next(): A = {
    i += 1
    seq(i - 1)
  }
}
