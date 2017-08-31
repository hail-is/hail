package is.hail.annotations

abstract class UnsafeOrdering extends Ordering[RegionValue] with Serializable {
  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int

  def compare(rv1: RegionValue, rv2: RegionValue): Int =
    compare(rv1.region, rv1.offset, rv2.region, rv2.offset)
}
