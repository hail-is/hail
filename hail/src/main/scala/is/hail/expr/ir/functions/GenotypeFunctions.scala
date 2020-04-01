package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.types.{coerce => _, _}
import is.hail.expr.ir._
import is.hail.expr.types.physical.{PArray, PCode, PFloat64, PIndexableCode, PInt32, PType}
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32, Type}

object GenotypeFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("gqFromPL", TArray(tv("N", "int32")), TInt32, (_: Type, _: PType) => PInt32()) { case (r, rt, (tPL: PArray, _pl: Code[Long])) =>
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

    registerCodeWithMissingness("dosage", TArray(tv("N", "float64")), TFloat64,  (_: Type, _: PType) => PFloat64()
    ) { case (r, rt, gp) =>
      EmitCode.fromI(r.mb) { cb =>
        gp.toI(cb).flatMap(cb) { case (gpc: PIndexableCode) =>
          val gpv = gpc.memoize(cb, "dosage_gp")

          cb.ifx(gpv.loadLength().cne(3),
            Code._fatal[Unit](const("length of gp array must be 3, got ").concat(gpv.loadLength().toS)))

          gpv.loadElement(cb, 1).flatMap(cb) { (_1: PCode) =>
            gpv.loadElement(cb, 2).map { (_2: PCode) =>
              PCode(rt, _1.tcode[Double] + _2.tcode[Double] * 2.0)
            }
          }
        }
      }
    }
  }
}
