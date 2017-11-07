package is.hail.sparkextras

import is.hail.annotations.{JoinedRegionValue, RegionValue, UnsafeOrdering}

abstract class OrderedJoinDistinctIterator(typ: OrderedRDD2Type, lrKOrd: UnsafeOrdering, leftIt: Iterator[RegionValue],
  rightIt: Iterator[RegionValue]) extends Iterator[JoinedRegionValue] {
  val jrv = JoinedRegionValue()
  var lrv: RegionValue = _
  var rrv: RegionValue = _

  var present = false
  var foundOne = false

  val rowType = typ.rowType
  var prev: WritableRegionValue = WritableRegionValue(typ.kType)

  val kRowOrd: UnsafeOrdering = typ.kRowOrd

  def advanceLeft() {
    assert(rrv != null)
    while (lrv != null && lrKOrd.compare(lrv, rrv) < 0) {
      if (leftIt.hasNext)
        lrv = leftIt.next()
      else
        lrv = null
    }
  }

  def advanceRight() {
    assert(lrv != null)
    while (rrv != null && lrKOrd.compare(lrv, rrv) > 0) {
      if (rightIt.hasNext)
        rrv = rightIt.next()
      else
        rrv = null
    }
  }

  def wasSeen(rv: RegionValue): Boolean = {
    if (foundOne) {
      kRowOrd.compare(rv, prev.value) == 0
    } else
      false
  }

  def hasNext(): Boolean

  def next(): JoinedRegionValue = {
    present = false
    jrv
  }
}

class OrderedInnerJoinDistinctIterator(typ: OrderedRDD2Type, lrKOrd: UnsafeOrdering, leftIt: Iterator[RegionValue],
  rightIt: Iterator[RegionValue]) extends OrderedJoinDistinctIterator(typ, lrKOrd, leftIt, rightIt) {

  def hasNext(): Boolean = {
    if (!present) {
      lrv = if (leftIt.hasNext) leftIt.next() else null
      rrv = if (rightIt.hasNext) rightIt.next() else null
    }

    while (!present && lrv != null && rrv != null) {
      val c = lrKOrd.compare(lrv, rrv)
      if (wasSeen(lrv)) {
        lrv = if (leftIt.hasNext) leftIt.next() else null
      } else if (c == 0) {
        jrv.set(lrv, rrv)
        prev.setSelect(rowType, Array(0, 1), lrv)
        foundOne = true
        present = true
      } else if (c > 0) {
        advanceRight()
      } else {
        advanceLeft()
      }
    }
    present
  }
}

class OrderedLeftJoinDistinctIterator(typ: OrderedRDD2Type, lrKOrd: UnsafeOrdering, leftIt: Iterator[RegionValue],
  rightIt: Iterator[RegionValue]) extends OrderedJoinDistinctIterator(typ, lrKOrd, leftIt, rightIt) {

  def hasNext(): Boolean = {
    if (!present) {
      lrv = if (leftIt.hasNext) leftIt.next() else null
      rrv = if (rightIt.hasNext) rightIt.next() else null
    }

    while (!present && lrv != null) {
      if (wasSeen(lrv)) {
        lrv = if (leftIt.hasNext) leftIt.next() else null
      } else if (rrv == null || lrKOrd.compare(lrv, rrv) <= 0) {
        jrv.set(lrv, rrv)
        prev.setSelect(rowType, Array(0, 1), lrv)
        foundOne = true
        present = true
      } else {
        advanceRight()
      }
    }
    present
  }
}
