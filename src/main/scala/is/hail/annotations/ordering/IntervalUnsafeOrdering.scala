package is.hail.annotations.ordering

import is.hail.variant.Contig
import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class IntervalUnsafeOrdering(t: TInterval, missingIsGreatest: Boolean) extends CodifiedUnsafeOrdering {
  private val representation = t.representation
  private val locusOrd = TLocus(t.gr).unsafeOrdering(missingIsGreatest)

  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
    val sOff1 = representation.loadField(r1, o1, 0)
    val sOff2 = representation.loadField(r2, o2, 0)

    val c1 = locusOrd.compare(r1, sOff1, r2, sOff2)
    if (c1 != 0)
      return c1

    val eOff1 = representation.loadField(r1, o1, 1)
    val eOff2 = representation.loadField(r2, o2, 1)

    locusOrd.compare(r1, eOff1, r2, eOff2)
  }

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    val out = fb.newLocal[Int]

    Code(
      out := locusOrd.compare(
        r1, representation.loadField(r1, o1, 0),
        r2, representation.loadField(r2, o2, 0))(fb, mb),
        out.cne(0).mux(
          out,
          locusOrd.compare(
            r1, representation.loadField(r1, o1, 1),
            r2, representation.loadField(r2, o2, 1))(fb, mb)))
  }
}
