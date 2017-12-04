package is.hail.annotations.ordering

import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

object DoubleUnsafeOrdering extends CodifiedUnsafeOrdering {
  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int =
    java.lang.Double.compare(r1.loadDouble(o1), r2.loadDouble(o2))

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    Code.invokeStatic[java.lang.Double, Double, Double, Int]("compare", r1.loadDouble(o1), r2.loadDouble(o2))
  }
}
