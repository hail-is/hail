package is.hail.annotations

import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

case class CodeOrdering(t: Type, missingGreatest: Boolean) {

  private[this] def booleanCompare(v1: Code[Boolean], v2: Code[Boolean]): Code[Int] =
    Code.invokeStatic[java.lang.Boolean, Boolean, Boolean, Int]("compare", v1, v2)

  private[this] def byteCompare(v1: Code[Byte], v2: Code[Byte]): Code[Int] =
    Code.invokeStatic[java.lang.Byte, Byte, Byte, Int]("compare", v1, v2)

  private[this] def intCompare(v1: Code[Int], v2: Code[Int]): Code[Int] =
    Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", v1, v2)

  private[this] def longCompare(v1: Code[Long], v2: Code[Long]): Code[Int] =
    Code.invokeStatic[java.lang.Long, Long, Long, Int]("compare", v1, v2)

  private[this] def floatCompare(v1: Code[Float], v2: Code[Float]): Code[Int] =
    Code.invokeStatic[java.lang.Float, Float, Float, Int]("compare", v1, v2)

  private[this] def doubleCompare(v1: Code[Double], v2: Code[Double]): Code[Int] =
    Code.invokeStatic[java.lang.Double, Double, Double, Int]("compare", v1, v2)

  def compare(mb: MethodBuilder, r1: Code[Region], o1: Code[Long], r2: Code[Region], o2: Code[Long]): Code[Int] =
    compare(t, mb, r1, o1, r2, o2)

  private[this] def compare(typ: Type, mb: MethodBuilder, r1: Code[Region], o1: Code[Long], r2: Code[Region], o2: Code[Long]): Code[Int] = {
    val c: Code[_] = typ match {
      case _: TBoolean =>
        booleanCompare(r1.loadBoolean(o1), r2.loadBoolean(o2))
      case _: TInt32 =>
        intCompare(r1.loadInt(o1), r2.loadInt(o2))
      case _: TInt64 =>
        longCompare(r1.loadLong(o1), r2.loadLong(o2))
      case _: TFloat32 =>
        floatCompare(r1.loadFloat(o1), r2.loadFloat(o2))
      case _: TFloat64 =>
        doubleCompare(r1.loadDouble(o1), r2.loadDouble(o2))
      case _: TBinary =>
        val b1 = mb.newLocal[Long]("bin_ord_b1")
        val b2 = mb.newLocal[Long]("bin_ord_b2")
        val l1 = mb.newLocal[Int]("bin_ord_l1")
        val l2 = mb.newLocal[Int]("bin_ord_l2")
        val lim = mb.newLocal[Int]("bin_ord_lim")
        val i = mb.newLocal[Int]("bin_ord_i")
        val cmp = mb.newLocal[Int]("bin_ord_cmp")
        Code(
          b1 := o1,
          b2 := o2,
          l1 := TBinary.loadLength(r1, o1),
          l2 := TBinary.loadLength(r2, o2),
          lim := (l1 < l2).mux(l1, l2),
          i := 0,
          cmp := 0,
          Code.whileLoop(cmp.ceq(0) && i < lim,
            cmp := byteCompare(r1.loadByte(TBinary.bytesOffset(b1) + i.toL),
              r2.loadByte(TBinary.bytesOffset(b2) + i.toL)),
            i += 1),
          cmp.ceq(0).mux(intCompare(l1, l2), cmp))
      case _: TLocus =>
      // FIXME: I think this needs GenomeRef stuff to work.
      case ti: TInterval =>
        val i1 = mb.newLocal[Long]
        val i2 = mb.newLocal[Long]
        val pdef1 = mb.newLocal[Boolean]
        val pinclude1 = mb.newLocal[Boolean]
        val cmp = mb.newLocal[Int]
        Code(
          i1 := o1,
          i2 := o2,
          pdef1 := ti.startDefined(r1, i1),
          pdef1.cne(ti.startDefined(r2, i2)).mux(
            if (missingGreatest) pdef1.mux(-1, 1) else pdef1.mux(1, -1),
            pdef1.mux(
              Code(
                cmp := compare(ti.pointType, mb, r1, ti.loadStart(r1, i1), r2, ti.loadStart(r2, i2)),
                cmp.cne(0).mux(cmp,
                  Code(
                    pinclude1 := ti.includeStart(r1, i1),
                    pinclude1.cne(ti.includeStart(r2, i2)).mux(
                      pinclude1.mux(-1, 1),
                      Code(
                        pdef1 := ti.endDefined(r1, i1),
                        pdef1.cne(ti.endDefined(r2, i2)).mux(
                          if (missingGreatest) pdef1.mux(-1, 1) else pdef1.mux(1, -1),
                          pdef1.mux(
                            Code(
                              cmp := compare(ti.pointType, mb, r1, ti.loadEnd(r1, i1), r2, ti.loadEnd(r2, i2)),
                              cmp.cne(0).mux(cmp,
                                Code(
                                  pinclude1 := ti.includeEnd(r1, i1),
                                  pinclude1.cne(ti.includeEnd(r1, i1))
                                    .mux(pinclude1.mux(1, -1), 0)))),
                            0))))))),
              0)))
      case ts: TBaseStruct =>
        var i = 0
        val leftDefined = mb.newLocal[Boolean]
        val rightDefined = mb.newLocal[Boolean]
        val cmp = mb.newLocal[Int]

        var compPrev = { c: Code[Unit] => c }
        while (i < ts.size) {
          val compareField = { cmpnextfield: Code[Unit] =>
            Code(
              leftDefined := ts.isFieldDefined(r1, o1, i),
              rightDefined := ts.isFieldDefined(r2, o2, i),
              (leftDefined && rightDefined).mux(
                cmp := compare(ts.types(i), mb, r1, o1, r2, o2),
                leftDefined.ceq(rightDefined).mux(
                  Code._empty,
                  cmp := (if (missingGreatest) leftDefined.mux(-1, 1) else leftDefined.mux(1, -1)))),
              cmp.ceq(0).mux(cmpnextfield, Code._empty))
          }
          compPrev = compPrev(compareField(_))
          i += 1
        }
        Code(cmp := 0, compPrev(Code._empty), cmp)

      case ti: TIterable =>
        val ftype = coerce[TArray](ti.fundamentalType)
        val etype = ftype.elementType
        val a1 = mb.newLocal[Long]("it_ord_a1")
        val a2 = mb.newLocal[Long]("it_ord_a2")
        val l1 = mb.newLocal[Int]("it_ord_l1")
        val l2 = mb.newLocal[Int]("it_ord_l2")
        val lim = mb.newLocal[Int]("it_ord_lim")
        val i = mb.newLocal[Int]("it_ord_i")
        val cmp = mb.newLocal[Int]("it_ord_cmp")
        val leftDefined = mb.newLocal[Boolean]
        val rightDefined = mb.newLocal[Boolean]
        Code(
          a1 := o1,
          a2 := o2,
          l1 := ftype.loadLength(r1, o1),
          l2 := ftype.loadLength(r2, o2),
          lim := (l1 < l2).mux(l1, l2),
          i := 0,
          cmp := 0,
          Code.whileLoop(cmp.ceq(0) && i < lim,
            leftDefined := ftype.isElementDefined(r1, a1, i),
            rightDefined := ftype.isElementDefined(r2, a2, i),
            (leftDefined && rightDefined).mux(
              cmp := compare(etype, mb,
                r1, ftype.loadElement(r1, a1, i),
                r2, ftype.loadElement(r2, a2, i)),
              leftDefined.ceq(rightDefined).mux(
                cmp := (if (missingGreatest) leftDefined.mux(-1, 1) else leftDefined.mux(1, -1)),
                Code._empty)),
            i += 1),
          cmp.ceq(0).mux(intCompare(l1, l2), cmp))
      case _ =>
        throw new UnsupportedOperationException(s"can't stage type: $t")
    }
    coerce[Int](c)
  }
}
