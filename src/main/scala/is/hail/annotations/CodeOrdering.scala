package is.hail.annotations

import java.io.PrintStream

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types._
import is.hail.utils._
import is.hail.expr.types.coerce
import is.hail.variant.ReferenceGenome

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

  def compare(mb: EmitMethodBuilder, v1: Code[_], v2: Code[_]): Code[Int] = {
    t match {
      case _: TBoolean => booleanCompare(asm4s.coerce[Boolean](v1), asm4s.coerce[Boolean](v2))
      case _: TInt32 | _: TCall => intCompare(asm4s.coerce[Int](v1), asm4s.coerce[Int](v2))
      case _: TInt64 => longCompare(asm4s.coerce[Long](v1), asm4s.coerce[Long](v2))
      case _: TFloat32 => floatCompare(asm4s.coerce[Float](v1), asm4s.coerce[Float](v2))
      case _: TFloat64 => doubleCompare(asm4s.coerce[Double](v1), asm4s.coerce[Double](v2))
      case _ =>
        compare(mb, mb.getArg[Region](1), asm4s.coerce[Long](v1), mb.getArg[Region](1), asm4s.coerce[Long](v2))
    }
  }

  def compare(mb: EmitMethodBuilder, r1: Code[Region], o1: Code[Long], r2: Code[Region], o2: Code[Long]): Code[Int] = {
    compare(t, mb, r1, o1, r2, o2)
  }

  private[this] def compare(typ: Type, mb: EmitMethodBuilder, r1: Code[Region], o1: Code[Long], r2: Code[Region], o2: Code[Long]): Code[Int] = {
    typ match {
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
      case _: TBinary | _: TString =>
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
          l1 := TBinary.loadLength(r1, b1),
          l2 := TBinary.loadLength(r2, b2),
          lim := (l1 < l2).mux(l1, l2),
          i := 0,
          cmp := 0,
          Code.whileLoop(cmp.ceq(0) && i < lim,
            cmp := byteCompare(r1.loadByte(TBinary.bytesOffset(b1) + i.toL),
              r2.loadByte(TBinary.bytesOffset(b2) + i.toL)),
            i += 1),
          cmp.ceq(0).mux(intCompare(l1, l2), cmp))

      case tl@TLocus(rg, _) =>
        val rgVal = mb.getReferenceGenome(rg.asInstanceOf[ReferenceGenome])
        val l1 = mb.newLocal[Long]
        val l2 = mb.newLocal[Long]
        val p1 = mb.newLocal[Long]("this_is_test1")
        val p2 = mb.newLocal[Long]("this_is_test2")
        val cmp = mb.newLocal[Int]

        val c1 = tl.representation.loadField(r1, l1, 0)
        val c2 = tl.representation.loadField(r2, l2, 0)

        val s1 = Code.invokeScalaObject[Region, Long, String](TString.getClass, "loadString", r1, c1)
        val s2 = Code.invokeScalaObject[Region, Long, String](TString.getClass, "loadString", r2, c2)

        Code(
          l1 := o1,
          l2 := o2,
          cmp := rgVal.invoke[String, String, Int]("compare", s1, s2),
          cmp.ceq(0).mux(Code(
            p1 := tl.representation.loadField(r1, l1, 1),
            p2 := tl.representation.loadField(r2, l2, 1),
            intCompare(r1.loadInt(p1), r2.loadInt(p2))), cmp))

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
                                  pinclude1.cne(ti.includeEnd(r2, i2))
                                    .mux(pinclude1.mux(1, -1), 0)))),
                            0))))))),
              0)))
      case ts: TBaseStruct =>
        var i = ts.size - 1
        val leftDefined = mb.newLocal[Boolean]
        val rightDefined = mb.newLocal[Boolean]
        val cmp = mb.newLocal[Int]

        var compNext = Code._empty[Unit]
        while(i >= 0) {
          compNext = Code(
              leftDefined := ts.isFieldDefined(r1, o1, i),
              rightDefined := ts.isFieldDefined(r2, o2, i),
              (leftDefined && rightDefined).mux(
                cmp := compare(ts.types(i), mb, r1, ts.loadField(r1, o1, i), r2, ts.loadField(r2, o2, i)),
                leftDefined.ceq(rightDefined).mux(
                  Code._empty,
                  cmp := (if (missingGreatest) leftDefined.mux(-1, 1) else leftDefined.mux(1, -1)))),
              cmp.ceq(0).mux(compNext, Code._empty))
          i -= 1
        }
        Code(cmp := 0, compNext, cmp)

      case tc: TContainer =>
        val etype = tc.elementType

        val a1 = mb.newLocal[Long]("it_ord_a1")
        val a2 = mb.newLocal[Long]("it_ord_a2")
        val l1 = mb.newLocal[Int]("it_ord_l1")
        val l2 = mb.newLocal[Int]("it_ord_l2")
        val lim = mb.newLocal[Int]("it_ord_lim")
        val i = mb.newLocal[Int]("it_ord_i")
        val cmp = mb.newLocal[Int]("it_ord_cmp")
        val leftDefined = mb.newLocal[Boolean]
        val rightDefined = mb.newLocal[Boolean]
        asm4s.coerce[Int](Code(
          a1 := o1,
          a2 := o2,
          l1 := tc.loadLength(r1, o1),
          l2 := tc.loadLength(r2, o2),
          lim := (l1 < l2).mux(l1, l2),
          i := 0,
          cmp := 0,
          Code.whileLoop(cmp.ceq(0) && i < lim,
            leftDefined := tc.isElementDefined(r1, a1, i),
            rightDefined := tc.isElementDefined(r2, a2, i),
            (leftDefined && rightDefined).mux(
              cmp := compare(etype, mb,
                r1, tc.loadElement(r1, a1, i),
                r2, tc.loadElement(r2, a2, i)),
              leftDefined.cne(rightDefined).mux(
                cmp := (if (missingGreatest) leftDefined.mux(-1, 1) else leftDefined.mux(1, -1)),
                Code._empty)),
            i += 1),
          cmp.ceq(0).mux(intCompare(l1, l2), cmp)))
      case tc: ComplexType =>
        compare(tc.representation, mb, r1, o1, r2, o2)
      case _ =>
        throw new UnsupportedOperationException(s"can't stage type: $typ")
    }
  }
}
