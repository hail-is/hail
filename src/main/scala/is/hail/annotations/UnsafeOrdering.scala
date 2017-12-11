package is.hail.annotations

import is.hail.utils._
import is.hail.asm4s._

trait UnsafeOrdering extends Ordering[RegionValue] with Serializable {
  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int

  def compare(rv1: RegionValue, rv2: RegionValue): Int =
    compare(rv1.region, rv1.offset, rv2.region, rv2.offset)

  def compare(rv1: RegionValue, r2: MemoryBuffer, o2: Long): Int =
    compare(rv1.region, rv1.offset, r2, o2)

  def compare(r1: MemoryBuffer, o1: Long, rv2: RegionValue): Int =
    compare(r1, o1, rv2.region, rv2.offset)
}


trait CodifiedUnsafeOrdering extends UnsafeOrdering {
  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long]): BindingCode[Int]
}
