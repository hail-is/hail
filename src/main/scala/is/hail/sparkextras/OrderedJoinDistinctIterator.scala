package is.hail.sparkextras

import is.hail.annotations.{JoinedRegionValue, Region, RegionValue, RegionValueBuilder}
import is.hail.rvd.OrderedRVDType
import scala.collection.mutable.ArrayBuffer
import scala.collection.{Iterator, Iterable, BufferedIterator}


abstract class OrderedJoinIterator(leftTyp: OrderedRVType, rightTyp: OrderedRVType, leftIt: Iterator[RegionValue],
  rightIt: Iterator[RegionValue]) extends Iterator[JoinedRegionValue] {
  private val jrv = JoinedRegionValue()
  private val rCache = Region()
  private val rCacheIdx = ArrayBuffer[Long]()
  private val rvb = new RegionValueBuilder(rCache)

  private var lrv: RegionValue = _
  private var rrv: RegionValue = if (rightIt.hasNext) rightIt.next() else null

  private var jrvPresent = false
  private def rCachePresent = false
  private var rCacheCur = -1
  private val lrKOrd = OrderedRVType.selectUnsafeOrdering(leftTyp.rowType, leftTyp.kRowFieldIdx, rightTyp.rowType, rightTyp.kRowFieldIdx)

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

abstract class OrderedJoinDistinctIterator(leftTyp: OrderedRVDType, rightTyp: OrderedRVDType, leftIt: Iterator[RegionValue],
  rightIt: Iterator[RegionValue]) extends Iterator[JoinedRegionValue] {
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

class OrderedInnerJoinDistinctIterator(leftTyp: OrderedRVDType, rightTyp: OrderedRVDType, leftIt: Iterator[RegionValue],
  rightIt: Iterator[RegionValue]) extends OrderedJoinDistinctIterator(leftTyp, rightTyp, leftIt, rightIt) {

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

class OrderedLeftJoinDistinctIterator(leftTyp: OrderedRVDType, rightTyp: OrderedRVDType, leftIt: Iterator[RegionValue],
  rightIt: Iterator[RegionValue]) extends OrderedJoinDistinctIterator(leftTyp, rightTyp, leftIt, rightIt) {

  def hasNext: Boolean = {
    if (!isPresent) {
      advanceLeft1()
      while (!isPresent && hasLeft) {
        if (!hasRight || lrCompare() < 0)
          setJRVRightNull()
        else if (lrCompare() == 0)
          setJRV()
        else
          advanceRight()
      }
    }

    isPresent
  }
}

class SteppedIterator[T](it: BufferedIterator[T], p: T => Boolean)
    extends Iterator[T] {

  private var stillValid = p

  override def takeWhile(p: T => Boolean): SteppedIterator[T] = {
    exhaust()
    stillValid = p
    this
  }

  def hasNext: Boolean = it.hasNext && stillValid(it.head)

  def next(): T = {
    if (!hasNext)
      throw new NoSuchElementException("next on empty iterator")
    it.next()
  }

  private def exhaust() {
    while (hasNext) next()
  }
}

//class OrderedGroupedIterator(
//  typ: OrderedRVType, it: BufferedIterator[RegionValue], ord: UnsafeOrdering)
//    extends Iterator[Iterable[RegionValue]] {
//
//  private val region = Region()
//  private val idx = ArrayBuffer[Long]()
//  private val rvb = new RegionValueBuilder(region)
//
//  def hasNext: Boolean = it.hasNext
//
//  def next(): Iterable[RegionValue] = {
//    if (!hasNext)
//      throw new NoSuchElementException("next on empty iterator")
//    region.clear()
//    idx.clear()
//  }
//
//}

//class OrderedCogroupedIterator(
//  leftTyp: OrderedRVType, rightTyp: OrderedRVType,
//  left: Iterator[RegionValue], right: Iterator[RegionValue])
//    extends Iterator[(Iterator[RegionValue], Iterable[RegionValue])] {
//
//  private val left = left.buffered
//  private val right = right.buffered
//
//  private val rStore = Region()
//  private val rStoreIdx = ArrayBuffer[Long]()
//  private val rStoreRvb = new RegionValueBuilder(rStore)
//
//  private val lrKOrd = OrderedRVType.selectUnsafeOrdering(
//    leftTyp.rowType, leftTyp.kRowFieldIdx,
//    rightTyp.rowType, rightTyp.kRowFieldIdx)
//
//  // TODO: Fix
//  private val leftPred: RegionValue => Boolean = _
//  private val lSteppedIt = new SteppedIterator(left, leftPred)
//
//  def lrCompare: Int = {
//    assert(left.hasNext && right.hasNext)
//    lrKOrd.compare(left.head, right.head)
//  }
//}
