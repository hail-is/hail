package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.types.physical.PArray
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32}
import is.hail.utils._
import is.hail.variant.Genotype

object GenotypeFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("gqFromPL", TArray(tv("N", "int32")), TInt32(), null) { case (r, rt, (tPL: PArray, pl: Code[Long])) =>
      val region = r.region
      val m = r.mb.newLocal[Int]("m")
      val m2 = r.mb.newLocal[Int]("m2")
      val len = r.mb.newLocal[Int]("len")
      val pli = r.mb.newLocal[Int]("pli")
      val i = r.mb.newLocal[Int]("i")
      Code(
        m := 99,
        m2 := 99,
        len := tPL.loadLength(region, pl),
        i := 0,
        Code.whileLoop(i < len,
          tPL.isElementDefined(region, pl, i).mux(
            Code._empty,
            Code._fatal("PL cannot have missing elements.")),
          pli := region.loadInt(tPL.loadElement(region, pl, len, i)),
          (pli < m).mux(
            Code(m2 := m, m := pli),
            (pli < m2).mux(
              m2 := pli,
              Code._empty)),
          i := i + 1
        ),
        m2 - m
      )
    }

    registerCode("dosage", TArray(tv("N", "float64")), TFloat64(), null) { case (r, rt, (pArray: PArray, gpOff: Code[Long])) =>
      val gp = r.mb.newLocal[Long]
      val region = r.region
      val len = pArray.loadLength(region, gp)

      Code(
        gp := gpOff,
        len.cne(3).mux(
          Code._fatal(const("length of gp array must be 3, got ").concat(len.toS)),
          region.loadDouble(pArray.elementOffset(gp, 3, 1)) +
            region.loadDouble(pArray.elementOffset(gp, 3, 2)) * 2.0))
    }

    // FIXME: remove when SkatSuite is moved to Python
    // the pl_dosage function in Python is implemented in Python
    registerCode("plDosage", TArray(tv("N", "int32")), TFloat64(), null) { case (r, rt, (pArray: PArray, plOff: Code[Long])) =>
      val pl = r.mb.newLocal[Long]
      val region = r.region
      val len = pArray.loadLength(region, pl)

      Code(
        pl := plOff,
        len.cne(3).mux(
          Code._fatal(const("length of pl array must be 3, got ").concat(len.toS)),
          Code.invokeScalaObject[Int, Int, Int, Double](Genotype.getClass, "plToDosage",
            region.loadInt(pArray.elementOffset(pl, 3, 0)),
            region.loadInt(pArray.elementOffset(pl, 3, 1)),
            region.loadInt(pArray.elementOffset(pl, 3, 2)))))
    }
  }
}