package is.hail.annotations.ordering

import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

object BooleanUnsafeOrdering extends CodifiedUnsafeOrdering {
  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int =
    java.lang.Boolean.compare(r1.loadBoolean(o1), r2.loadBoolean(o2))

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    Code.invokeStatic[java.lang.Boolean, Boolean, Boolean, Int]("compare", r1.loadBoolean(o1), r2.loadBoolean(o2))
  }
}
