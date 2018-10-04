package is.hail.utils.richUtils

/** Rich wrapper for an indexed sequence.
  *
  * Houses the generic binary search methods. All methods taking
  *   - a search key 'x: U',
  *   - a key comparison 'lt: (U, U) => Boolean',
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
  def lowerBound[U](x: U, start: Int, end: Int, lt: (U, U) => Boolean, k: (T) => U): Int =
    partitionPoint[U](!lt(_: U, x), start, end, k)

  def lowerBound[U >: T](x: U, start: Int, end: Int, lt: (U, U) => Boolean): Int =
    lowerBound(x, start, end, lt, identity[U])

  def lowerBound[U](x: U, lt: (U, U) => Boolean, k: (T) => U): Int =
    lowerBound(x, 0, a.length, lt, k)

  def lowerBound[U >: T](x: U, lt: (U, U) => Boolean): Int =
    lowerBound(x, 0, a.length, lt)

  /** Returns i in ['start', 'end'] such that
    *   - !('x' < a(i)) (i.e. a(i) <= 'x') for all i in ['start', i), and
    *   - 'x' < a(i) for all i in [i, 'end')
    */
  def upperBound[U](x: U, start: Int, end: Int, lt: (U, U) => Boolean, k: (T) => U): Int =
    partitionPoint[U](lt(x, _: U), start, end, k)

  def upperBound[U >: T](x: U, start: Int, end: Int, lt: (U, U) => Boolean): Int =
    upperBound(x, start, end, lt, identity[U])

  def upperBound[U](x: U, lt: (U, U) => Boolean, k: (T) => U): Int =
    upperBound(x, 0, a.length, lt, k)

  def upperBound[U >: T](x: U, lt: (U, U) => Boolean): Int =
    upperBound(x, 0, a.length, lt)

  /** Returns (l, u) such that
    *   - a(i) < 'x' for all i in [0, l),
    *   - !(a(i) < 'x') && !('x' < a(i)) (i.e. a(i) == 'x') for all i in [l, u),
    *   - 'x' < a(i) for all i in [u, a.size).
    */
  def equalRange[U](x: U, lt: (U, U) => Boolean, k: (T) => U): (Int, Int) =
    runSearch(x, lt, k,
      (l, m, u) =>
        (lowerBound(x, l, m, lt, k), upperBound(x, m + 1, u, lt, k)),
      (m) =>
        (m, m))

  def equalRange[U >: T](x: U, lt: (U, U) => Boolean): (Int, Int) =
    equalRange(x, lt, identity[U])

  def containsOrdered[U](x: U, lt: (U, U) => Boolean, k: (T) => U): Boolean =
    runSearch(x, lt, k, (_, _, _) => true, (_) => false)

  def containsOrdered[U >: T](x: U, lt: (U, U) => Boolean): Boolean =
    containsOrdered(x, lt, identity[U])

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
      val mid = (left + right) / 2
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
  private def runSearch[U, V](
    x: U,
    lt: (U, U) => Boolean,
    k: (T) => U,
    found: (Int, Int, Int) => V,
    notFound: (Int) => V
  ): V = {
    var left = 0
    var right = a.size
    while (left < right) {
      // a(i) < x for all i in [0, left)
      // x < a(i) for all i in [right, a.size)
      val mid = (left + right) / 2
      if (lt(x, k(a(mid))))
        // x < a(i) for all i in [mid, a.size)
        right = mid
      else if (lt(k(a(mid)), x))
        // a(i) < x for all i in [0, mid]
        left = mid + 1
      else
        // !(a(i) < x) for all i in [mid, a.size)
        // !(x < a(i)) for all i in [0, mid]
        return found(left, mid, right)
    }
    notFound(left)
  }
}
