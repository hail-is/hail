package is.hail.annotations.ordering

import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._

object BinaryUnsafeOrdering extends CodifiedUnsafeOrdering {
  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int =
    StaticBinaryUnsafeOrdering.compare(r1, o1, r2, o2)

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    Code.invokeStatic[StaticBinaryUnsafeOrdering, MemoryBuffer, Long, MemoryBuffer, Long, Int]("compare", r1, o1, r2, o2)
  }
}
