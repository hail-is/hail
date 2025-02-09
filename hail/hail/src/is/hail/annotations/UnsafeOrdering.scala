package is.hail.annotations

abstract class UnsafeOrdering extends Ordering[Long] with Serializable {
  def compare(o1: Long, o2: Long): Int

  def compare(rv1: RegionValue, rv2: RegionValue): Int =
    compare(rv1.offset, rv2.offset)

  def compare(rv1: RegionValue, r2: Region, o2: Long): Int =
    compare(rv1.offset, o2)

  def compare(r1: Region, o1: Long, rv2: RegionValue): Int =
    compare(o1, rv2.offset)

  def toRVOrdering: Ordering[RegionValue] = on[RegionValue](rv => rv.offset)
}
