package is.hail.annotations.ordering

import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

object IntegerUnsafeOrdering extends CodifiedUnsafeOrdering {
  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int =
    Integer.compare(r1.loadInt(o1), r2.loadInt(o2))

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    Code.invokeStatic[Integer, Int, Int, Int]("compare", r1.loadInt(o1), r2.loadInt(o2))
  }
}
