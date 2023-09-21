package is.hail.utils.richUtils

import is.hail.utils._

import scala.reflect.ClassTag

class RichIndexedSeqAnyRef[T <: AnyRef](val a: IndexedSeq[T]) extends AnyVal {
  def elementsSameObjects(b: IndexedSeq[T]): Boolean = {
    if (a.length != b.length) return false
    var same = true
    var i = 0
    while (same && i < a.length) {
      same = a(i) eq b(i)
      i += 1
    }
    same
  }
}
/** Rich wrapper for an indexed sequence.
  *
  * Houses the generic binary search methods. All methods taking
  *   - a search key 'x: U',
  *   - a key comparison 'lt: (U, U) => Boolean' (the most generic versions
  *   allow the search key 'x' to be of a different type than the elements of
  *   the sequence, and take one or two mixed type comparison functions),
  *   - and a key projection 'k: (T) => U',
  * assume the following preconditions for all 0 <= i <= j < a.size (writing <
  * for 'lt'):
  *   1. if 'x' < k(a(i)) then 'x' < k(a(j))
  *   2. if k(a(j)) < 'x' then k(a(i)) < 'x'
  * These can be rephrased as 1: 'x' < k(_) partitions a, and 2: k(_) < 'x'
  * partitions a. (Actually, upperBound only needs 1. and lowerBound only needs
  * 2.)
  */
class RichIndexedSeq[T](val a: IndexedSeq[T]) extends AnyVal {

  /** Returns 'start' <= i <= 'end' such that
    *   - a(i) < 'x' for all i in ['start', i), and
    *   - !(a(i) < 'x') (i.e. 'x' <= a(i)) for all i in [i, 'end')
    */
  def lowerBound[U, V](x: V, start: Int, end: Int, lt: (U, V) => Boolean, k: (T) => U): Int =
    partitionPoint[U](!lt(_: U, x), start, end, k)

  def lowerBound[U >: T, V](x: V, start: Int, end: Int, lt: (U, V) => Boolean): Int =
    lowerBound(x, start, end, lt, identity[U])

  def lowerBound[U, V](x: V, lt: (U, V) => Boolean, k: (T) => U): Int =
    lowerBound(x, 0, a.length, lt, k)

  def lowerBound[U >: T, V](x: V, lt: (U, V) => Boolean): Int =
    lowerBound(x, 0, a.length, lt)

  /** Returns i in ['start', 'end'] such that
    *   - !('x' < a(i)) (i.e. a(i) <= 'x') for all i in ['start', i), and
    *   - 'x' < a(i) for all i in [i, 'end')
    */
  def upperBound[U, V](x: V, start: Int, end: Int, lt: (V, U) => Boolean, k: (T) => U): Int =
    partitionPoint[U](lt(x, _: U), start, end, k)

  def upperBound[U >: T, V](x: V, start: Int, end: Int, lt: (V, U) => Boolean): Int =
    upperBound(x, start, end, lt, identity[U])

  def upperBound[U, V](x: V, lt: (V, U) => Boolean, k: (T) => U): Int =
    upperBound(x, 0, a.length, lt, k)

  def upperBound[U >: T, V](x: V, lt: (V, U) => Boolean): Int =
    upperBound(x, 0, a.length, lt)

  /** Returns (l, u) such that
    *   - a(i) < 'x' for all i in [0, l),
    *   - !(a(i) < 'x') && !('x' < a(i)) (i.e. a(i) == 'x') for all i in [l, u),
    *   - 'x' < a(i) for all i in [u, a.size).
    */
  def equalRange[U](x: U, lt: (U, U) => Boolean, k: (T) => U): (Int, Int) =
    equalRange(x, lt, lt, k)

  def equalRange[U, V](
    x: V,
    ltUV: (U, V) => Boolean,
    ltVU: (V, U) => Boolean,
    k: (T) => U
  ): (Int, Int) =
    runSearch(x, ltUV, ltVU, k,
      (l, m, u) =>
        (lowerBound(x, l, m, ltUV, k), upperBound(x, m + 1, u, ltVU, k)),
      (m) =>
        (m, m))

  def equalRange[U >: T](x: U, lt: (U, U) => Boolean): (Int, Int) =
    equalRange(x, lt, lt, identity[U])

  def equalRange[U >: T, V](x: V, ltUV: (U, V) => Boolean, ltVU: (V, U) => Boolean): (Int, Int) =
    equalRange(x, ltUV, ltVU, identity[U])

  def containsOrdered[U, V](
    x: V,
    ltUV: (U, V) => Boolean,
    ltVU: (V, U) => Boolean,
    k: (T) => U
  ): Boolean = runSearch(x, ltUV, ltVU, k, (_, _, _) => true, (_) => false)

  def containsOrdered[U](x: U, lt: (U, U) => Boolean, k: (T) => U): Boolean =
    containsOrdered(x, lt, lt, k)

  def containsOrdered[U >: T](x: U, lt: (U, U) => Boolean): Boolean =
    containsOrdered(x, lt, lt, identity[U])

  def containsOrdered[U >: T, V](x: V, ltUV: (U, V) => Boolean, ltVU: (V, U) => Boolean): Boolean =
    containsOrdered(x, ltUV, ltVU, identity[U])

  /** Returns 'start' <= i <= 'end' such that p(k(a(j))) is false for all j
    * in ['start', i), and p(k(a(j))) is true for all j in [i, 'end').
    *
    * Assumes p(k(_)) partitions a, i.e. for all 0 <= i <= j < a.size,
    * if p(k(a(i))) then p(k(a(j))).
    */
  def partitionPoint[U](p: (U) => Boolean, start: Int, end: Int, k: (T) => U): Int = {
    var left = start
    var right = end
    while (left < right) {
      val mid = (left + right) >>> 1 // works even when sum overflows
      if (p(k(a(mid))))
        right = mid
      else
        left = mid + 1
    }
    left
  }

  def partitionPoint[U >: T](p: (U) => Boolean, start: Int, end: Int): Int =
    partitionPoint(p, start, end, identity[U])

  def partitionPoint[U](p: (U) => Boolean, k: (T) => U): Int =
    partitionPoint(p, 0, a.length, k)

  def partitionPoint[U >: T](p: (U) => Boolean): Int =
    partitionPoint(p, identity[U])

  /** Perform binary search until either an index i is found for which k(a(i))
    * is incomparible with 'x', or it is certain that no such i exists. In the
    * first case, call 'found'(l, i, u), where [l, u] is the current range of
    * the search. In the second case, call 'notFound'(j), where k(a(i)) < x for
    * all i in [0, j) and x < k(a(i)) for all i in [j, a.size).
    */
  private def runSearch[U, V, R](
    x: V,
    ltUV: (U, V) => Boolean,
    ltVU: (V, U) => Boolean,
    k: (T) => U,
    found: (Int, Int, Int) => R,
    notFound: (Int) => R
  ): R = {
    var left = 0
    var right = a.size
    while (left < right) {
      // a(i) < x for all i in [0, left)
      // x < a(i) for all i in [right, a.size)
      val mid = (left + right) >>> 1 // works even when sum overflows
      if (ltVU(x, k(a(mid))))
        // x < a(i) for all i in [mid, a.size)
        right = mid
      else if (ltUV(k(a(mid)), x))
        // a(i) < x for all i in [0, mid]
        left = mid + 1
      else
        // !(a(i) < x) for all i in [mid, a.size)
        // !(x < a(i)) for all i in [0, mid]
        return found(left, mid, right)
    }
    notFound(left)
  }

  def treeReduce(f: (T, T) => T)(implicit tct: ClassTag[T]): T = {
    var is: IndexedSeq[T] = a
    while (is.length > 1) {
      is = is.iterator.grouped(2).map {
        case Seq(x1, x2) => f(x1, x2)
        case Seq(x1) => x1
      }.toFastSeq
    }
    is.head
  }
}
