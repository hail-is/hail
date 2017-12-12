package is.hail.annotations

abstract class UnsafeOrdering extends Ordering[RegionValue] with Serializable {
  def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int

  def compare(rv1: RegionValue, rv2: RegionValue): Int =
    compare(rv1.region, rv1.offset, rv2.region, rv2.offset)

  def compare(rv1: RegionValue, r2: Region, o2: Long): Int =
    compare(rv1.region, rv1.offset, r2, o2)

  def compare(r1: Region, o1: Long, rv2: RegionValue): Int =
    compare(r1, o1, rv2.region, rv2.offset)
}
