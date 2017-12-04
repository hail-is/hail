package is.hail.annotations.ordering

import is.hail.variant.Contig
import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class LocusUnsafeOrdering(t: TLocus, missingIsGreatest: Boolean) extends CodifiedUnsafeOrdering {
  private val repr = t.representation.fundamentalType

  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
    val cOff1 = repr.loadField(r1, o1, 0)
    val cOff2 = repr.loadField(r2, o2, 0)

    val contig1 = TString.loadString(r1, cOff1)
    val contig2 = TString.loadString(r2, cOff2)

    val c = Contig.compare(contig1, contig2)
    if (c != 0)
      return c

    val posOff1 = repr.loadField(r1, o1, 1)
    val posOff2 = repr.loadField(r2, o2, 1)
    Integer.compare(r1.loadInt(posOff1), r2.loadInt(posOff2))
  }

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    val out = fb.newLocal[Int]

    Code(
      out := Code.invokeStatic[StaticContigUnsafeOrdering, String, String, Int]("compare",
        TString.loadString(r1, repr.loadField(r1, o1, 0)),
        TString.loadString(r2, repr.loadField(r2, o2, 0))),
      out.cne(0).mux(
        out,
        Code.invokeStatic[Integer, Int, Int, Int]("compare",
          r1.loadInt(repr.loadField(r1, o1, 1)),
          r2.loadInt(repr.loadField(r2, o2, 1)))))
  }
}
