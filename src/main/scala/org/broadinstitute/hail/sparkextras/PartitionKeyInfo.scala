package org.broadinstitute.hail.sparkextras

case class PartitionKeyInfo[T](
  partIndex: Int,
  sortedness: Int,
  min: T,
  max: T)

object PartitionKeyInfo {
  final val UNSORTED = 0
  final val TSORTED = 1
  final val KSORTED = 2

  def apply[T, K](partIndex: Int, projectKey: (K) => T, it: Iterator[K])(implicit tOrd: Ordering[T], kOrd: Ordering[K]): PartitionKeyInfo[T] = {
    import Ordering.Implicits._

    assert(it.hasNext)

    val k0 = it.next()
    val t0 = projectKey(k0)

    var minT = t0
    var maxT = t0
    var sortedness = KSORTED
    var prevK = k0
    var prevT = t0

    while (it.hasNext) {
      val k = it.next()
      val t = projectKey(k)

      if (t < prevT)
        sortedness = UNSORTED
      else if (k < prevK)
        sortedness = sortedness.min(TSORTED)

      if (t < minT)
        minT = t
      if (t > maxT)
        maxT = t

      prevK = k
      prevT = t
    }

    PartitionKeyInfo(partIndex, sortedness, minT, maxT)
  }
}