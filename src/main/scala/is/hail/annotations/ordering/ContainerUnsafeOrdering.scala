package is.hail.annotations.ordering

import is.hail.asm4s._
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._

class ContainerUnsafeOrdering(t: TContainer, eltOrd: CodifiedUnsafeOrdering, missingIsGreatest: Boolean) extends CodifiedUnsafeOrdering {
  override def compare(r1: MemoryBuffer, o1: Long, r2: MemoryBuffer, o2: Long): Int = {
    val length1 = t.loadLength(r1, o1)
    val length2 = t.loadLength(r2, o2)

    var i = 0
    while (i < math.min(length1, length2)) {
      val leftDefined = t.isElementDefined(r1, o1, i)
      val rightDefined = t.isElementDefined(r2, o2, i)

      if (leftDefined && rightDefined) {
        val eOff1 = t.loadElement(r1, o1, length1, i)
        val eOff2 = t.loadElement(r2, o2, length2, i)
        val c = eltOrd.compare(r1, eOff1, r2, eOff2)
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
    Integer.compare(length1, length2)
  }

  def compare(r1: Code[MemoryBuffer], o1: Code[Long], r2: Code[MemoryBuffer], o2: Code[Long])
      : BindingCode[Int] = { (fb, mb) =>
    val length1 = fb.newLocal[Int]
    val length2 = fb.newLocal[Int]
    val i = fb.newLocal[Int]
    val min = fb.newLocal[Int]
    val leftDefined = mb.newBit
    val rightDefined = mb.newBit
    val temp = fb.newLocal[Int]
    val out = fb.newLocal[Int]

    Code(
      length1 := t.loadLength(r1, o1),
      length2 := t.loadLength(r2, o2),
      i := 0,
      min := Code.invokeStatic[java.lang.Math, Int, Int, Int]("min", length1, length2),
      out := 0,
      Code.whileLoop(i < min,
        leftDefined := t.isElementDefined(r1, o1, i),
        rightDefined := t.isElementDefined(r2, o2, i),
        (leftDefined & rightDefined & out.cne(0)).mux(
          Code(
            temp := eltOrd.compare(
              r1, t.loadElement(r1, o1, length1, i),
              r2, t.loadElement(r2, o2, length2, i))(fb, mb),
            temp.cne(0).mux(Code._empty, out := temp)),
          (leftDefined ^ rightDefined).mux(
            out :=
              (if (missingIsGreatest) leftDefined.mux(-1, 1)
              else leftDefined.mux(1, -1)),
            Code._empty)),
        i += 1
      ),
      out.cne(0).mux(out, Code.invokeStatic[Integer, Int, Int, Int]("compare", length1, length2)))
  }
}
