package is.hail.annotations.ordering

import is.hail.variant._
import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class VariantUnsafeOrdering(t: TVariant, missingIsGreatest: Boolean) extends CodifiedUnsafeOrdering {
  private val fundamentalComparators = t.representation.fields.map(_.typ.unsafeOrdering(missingIsGreatest)).toArray
  private val repr = t.representation.fundamentalType

  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
    val cOff1 = repr.loadField(r1, o1, 0)
    val cOff2 = repr.loadField(r2, o2, 0)

    val contig1 = TString.loadString(r1, cOff1)
    val contig2 = TString.loadString(r2, cOff2)

    val c = Contig.compare(contig1, contig2)
    if (c != 0)
      return c

    var i = 1
    while (i < repr.size) {
      val fOff1 = repr.loadField(r1, o1, i)
      val fOff2 = repr.loadField(r2, o2, i)

      val c = fundamentalComparators(i).compare(r1, fOff1, r2, fOff2)
      if (c != 0)
        return c

      i += 1
    }
    0
  }

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    val out = fb.newLocal[Int]
    val i = fb.newLocal[Int]

    Code(
      out := Code.invokeStatic[StaticContigUnsafeOrdering, String, String, Int]("compare",
        TString.loadString(r1, repr.loadField(r1, o1, 0)),
        TString.loadString(r2, repr.loadField(r2, o2, 0))),
      out.cne(0).mux(
        out,
        (0 until repr.size).foldRight[Code[Int]](0) { (i, subsequentChecks) =>
          Code(
            fundamentalComparators(i).compare(r1, repr.loadField(r1, o1, i), r2, repr.loadField(r2, o2, i))(fb, mb),
            out.cne(0).mux(out, subsequentChecks)) }))
  }
}
