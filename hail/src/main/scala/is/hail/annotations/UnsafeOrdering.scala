package is.hail.annotations

abstract class UnsafeOrdering extends Ordering[RegionValue] with Serializable {
  def compare(o1: Long, o2: Long): Int

  def compare(rv1: RegionValue, rv2: RegionValue): Int =
    compare(rv1.offset, rv2.offset)
}
