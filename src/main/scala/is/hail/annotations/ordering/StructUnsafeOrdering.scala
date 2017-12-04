package is.hail.annotations.ordering

import is.hail.variant.Contig
import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class StructUnsafeOrdering(private val t: TStruct, missingIsGreatest: Boolean) extends CodifiedUnsafeOrdering {
  private val fieldOrderings = t.fields.map(_.typ.unsafeOrdering(missingIsGreatest)).toArray

  def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
    var i = 0
    while (i < t.size) {
      val leftDefined = t.isFieldDefined(r1, o1, i)
      val rightDefined = t.isFieldDefined(r2, o2, i)

      if (leftDefined && rightDefined) {
        val c = fieldOrderings(i).compare(r1, t.loadField(r1, o1, i), r2, t.loadField(r2, o2, i))
        if (c != 0)
          return c
      } else if (leftDefined != rightDefined) {
        val c = if (leftDefined) -1 else 1
        if (missingIsGreatest)
          return c
        else
          return -c
      }
      i += 1
    }
    0
  }

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    val out = fb.newLocal[Int]
    val leftDefined = mb.newBit
    val rightDefined = mb.newBit

    (0 until t.size).foldRight[Code[Int]](0) { (i, checkSubsequentFields) =>
      Code(
        leftDefined := t.isFieldDefined(r1, o1, i),
        rightDefined := t.isFieldDefined(r2, o2, i),
        out :=
          (leftDefined && rightDefined).mux(
            fieldOrderings(i).compare(r1, t.loadField(r1, o1, i), r2, t.loadField(r2, o2, i))(fb, mb),
            (leftDefined ^ rightDefined).mux(
              if (missingIsGreatest) leftDefined.mux(-1, 1) else leftDefined.mux(1, -1),
              0)),
        out.cne(0).mux(out, checkSubsequentFields))
    }
  }
}
