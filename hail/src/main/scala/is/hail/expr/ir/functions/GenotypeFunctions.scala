package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.types.{coerce => _, _}
import is.hail.expr.ir._
import is.hail.expr.types.physical.PArray
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32}
import is.hail.variant.Genotype

object GenotypeFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("gqFromPL", TArray(tv("N", "int32")), TInt32, null) { case (r, rt, (tPL: PArray, _pl: Code[Long])) =>
      val pl = r.mb.newLocal[Long]("pl")
      val m = r.mb.newLocal[Int]("m")
      val m2 = r.mb.newLocal[Int]("m2")
      val len = r.mb.newLocal[Int]("len")
      val pli = r.mb.newLocal[Int]("pli")
      val i = r.mb.newLocal[Int]("i")
      Code(
        pl := _pl,
        m := 99,
        m2 := 99,
        len := tPL.loadLength(pl),
        i := 0,
        Code.whileLoop(i < len,
          tPL.isElementDefined(pl, i).mux(
            Code._empty,
            Code._fatal[Unit]("PL cannot have missing elements.")),
          pli := Region.loadInt(tPL.loadElement(pl, len, i)),
          (pli < m).mux(
            Code(m2 := m, m := pli),
            (pli < m2).mux(
              m2 := pli,
              Code._empty)),
          i := i + 1
        ),
        m2 - m)
    }

    registerCode[Long]("dosage", TArray(tv("N", "float64")), TFloat64, null) { case (r, rt, (gpPType, gpOff)) =>
      val gpPArray = coerce[PArray](gpPType)

      Code.memoize(gpOff, "dosage_gp") { gp =>
        Code.memoize(gpPArray.loadLength(gp), "dosage_len") { len =>
        len.cne(3).mux(
          Code._fatal[Double](const("length of gp array must be 3, got ").concat(len.toS)),
          Region.loadDouble(gpPArray.elementOffset(gp, 3, 1)) +
            Region.loadDouble(gpPArray.elementOffset(gp, 3, 2)) * 2.0)
        }
      }
    }

    // FIXME: remove when SkatSuite is moved to Python
    // the pl_dosage function in Python is implemented in Python
    registerCode[Long]("plDosage", TArray(tv("N", "int32")), TFloat64, null) { case (r, rt, (plPType, plOff)) =>
      val plPArray = coerce[PArray](plPType)

      Code.memoize(plOff, "plDosage_pl") { pl =>
        Code.memoize(plPArray.loadLength(pl), "plDosage_len") { len =>
          len.cne(3).mux(
            Code._fatal[Double](const("length of pl array must be 3, got ").concat(len.toS)),
            Code.invokeScalaObject[Int, Int, Int, Double](Genotype.getClass, "plToDosage",
              Region.loadInt(plPArray.elementOffset(pl, 3, 0)),
              Region.loadInt(plPArray.elementOffset(pl, 3, 1)),
              Region.loadInt(plPArray.elementOffset(pl, 3, 2))))
        }
      }
    }
  }
}