package is.hail.utils

import scala.collection.mutable
import scala.reflect.ClassTag

class BinaryHeap[@specialized T : ClassTag](initialCapacity: Int = 32) {
  private class RankedT(val v: T, var rank: Long) {}
  private var a: Array[RankedT] = new Array[RankedT](initialCapacity)
  private val m: mutable.Map[T, Int] = new mutable.HashMap()
  private var next: Int = 0

  def size: Int = next

  def insert(t: T, r: Long) {
    if (m.contains(t))
      throw new RuntimeException(s"key $t already exists with priority ${m(t)}, cannot add it again with priority $r")
    maybeGrow()
    put(next, new RankedT(t,r))
    bubbleUp(next)
    next += 1
  }

  def maxWithPriority(): (T, Long) = {
    val rt = maxRanked()
    (rt.v, rt.rank)
  }

  def max(): T =
    maxRanked().v

  def maxPriority(): Long =
    maxRanked().rank

  def extractMax(): T = {
    val max = maxRanked().v
    next -= 1
    m -= max
    if (next > 0) {
      put(0, a(next))
      bubbleDown(0)
      maybeShrink()
    }
    max
  }

  def getPriority(t: T): Long =
    a(m(t)).rank

  def decreasePriorityTo(t: T, r: Long) {
    val i = m(t)
    assert(a(i).rank > r)
    a(i).rank = r
    bubbleDown(i)
  }

  def increasePriorityTo(t: T, r: Long) {
    val i = m(t)
    assert(a(i).rank < r)
    a(i).rank = r
    bubbleUp(i)
  }

  def contains(t: T): Boolean =
    m.contains(t)

  def toArray: Array[T] = {
    val trimmed = new Array(size)
    var i = 0
    while (i < size) {
      trimmed(i) = a(i).v
      i += 1
    }
    trimmed
  }

  private def maxRanked(): RankedT =
    if (next > 0)
      a(0)
    else
      throw new RuntimeException("heap is empty")

  private def parent(i: Int) =
    if (i == 0) 0 else (i - 1) >>> 1

  private def put(to: Int, rt: RankedT) {
    a(to) = rt
    m(rt.v) = to
  }

  private def swap(i: Int, j: Int) {
    val temp = a(i)
    a(i) = a(j)
    a(j) = temp
    m(a(j).v) = j
    m(a(i).v) = i
  }

  private def maybeGrow() {
    if (next >= a.length) {
      val a2 = new Array[RankedT](a.length << 1)
      var j = 0
      while (j < a.length) {
        a2(j) = a(j)
        j += 1
      }
      a = a2
    }
  }

  private def maybeShrink() {
    if (next < (a.length >>> 2)) {
      val a2 = new Array[RankedT](a.length >>> 2)
      var j = 0
      while (j < a2.length) {
        a2(j) = a(j)
        j += 1
      }
      a = a2
    }
  }

  private def bubbleUp(i: Int) {
    var current = i
    var p = parent(current)
    while (a(current).rank > a(p).rank) {
      swap(current, p)
      current = p
      p = parent(current)
    }
  }

  private def bubbleDown(i: Int) {
    var current = i
    var largest = current
    var continue = false
    do {
      val leftChild = (current << 1) + 1
      val rightChild = (current << 1) + 2

      if (leftChild < next && a(leftChild).rank > a(largest).rank)
        largest = leftChild
      if (rightChild < next && a(rightChild).rank > a(largest).rank)
        largest = rightChild

      if (largest != current) {
        swap(largest, current)
        largest = current
        continue = true
      } else
        continue = false
    } while (continue);
  }

}
