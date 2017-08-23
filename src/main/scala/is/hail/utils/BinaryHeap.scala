package is.hail.utils

import scala.collection.mutable
import scala.reflect.ClassTag

class BinaryHeap[@specialized T : ClassTag](initialCapacity: Int = 32) extends PriorityQueue[T, Long] {
  private class RankedT(val v: T, val rank: Long) {}
  private var a: Array[RankedT] = new Array[RankedT](initialCapacity)
  private val m: mutable.Map[T, Int] = new mutable.HashMap()
  private var next: Int = 0

  private def parent(i: Int) = if (i == 0) 0 else (i - 1) >>> 1

  def size: Int = next

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

  private def putNext(rt: RankedT) {
    if (next > a.length) {
      val a2 = new Array[RankedT](a.length << 1)
      var j = 0
      while (j < a.length) {
        a2(j) = a(j)
        j += 1
      }
      a = a2
    }
    put(next, rt)
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

  def insert(t: T, r: Long) {
    if (m.contains(t))
      throw new RuntimeException(s"key $t already exists with priority ${m(t)}, cannot add it again with priority $r; use setPriority")
    putNext(new RankedT(t,r))
    bubbleUp(next)
    next += 1
  }

  private def maxWithPriority(): RankedT =
    if (next > 0)
      a(0)
    else
      throw new RuntimeException("heap is empty")

  def max(): T =
    maxWithPriority().v

  def maxPriority(): Long =
    maxWithPriority().rank

  def extractMax(): T = {
    val max = a(0).v
    next -= 1
    m.remove(max)
    put(0, a(next))
    maybeShrink()
    bubbleDown(0)
    max
  }

  def getPriority(t: T): Long =
    a(m(t)).rank

  def setPriority(t: T, r: Long) {
    changePriority(t, _ => r)
  }

  def changePriority(t: T, f: (Long) => Long) {
    val i = m(t)
    val oldR = a(i).rank
    val r = f(oldR)
    if (r > oldR) {
      a(i) = new RankedT(t,r)
      bubbleUp(i)
    } else if (r < oldR) {
      a(i) = new RankedT(t,r)
      bubbleDown(i)
    }
  }

  def contains(t: T): Boolean =
    m.contains(t)

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

  def toArray: Array[T] = {
    val trimmed = new Array(size)
    var i = 0
    while (i < size) {
      trimmed(i) = a(i).v
      i += 1
    }
    trimmed
  }

}
