package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.types.{coerce => _, _}
import is.hail.expr.ir._
import is.hail.types.physical.{PArray, PCode, PFloat64, PIndexableCode, PInt32, PType}
import is.hail.types.virtual.{TArray, TFloat64, TInt32, Type}

object GenotypeFunctions extends RegistryFunctions {

  def registerAll() {
    registerPCode1("gqFromPL", TArray(tv("N", "int32")), TInt32, (_: Type, _: PType) => PInt32())
    { case (r, rt, _pl: PIndexableCode) =>
      val code = EmitCodeBuilder.scopedCode(r.mb) { cb =>
        val pl = _pl.memoize(cb, "plv")
        val m = cb.newLocal[Int]("m", 99)
        val m2 = cb.newLocal[Int]("m2", 99)
        val i = cb.newLocal[Int]("i", 0)

        cb.whileLoop(i < pl.loadLength(), {
          val value = pl.loadElement(cb, i)
            .handle(cb, cb += Code._fatal[Unit]("PL cannot have missing elements."))
          val pli = cb.newLocal[Int]("pli", value.tcode[Int])
          cb.ifx(pli < m, {
            cb.assign(m2, m)
            cb.assign(m, pli)
          }, {
            cb.ifx(pli < m2,
              cb.assign(m2, pli))
          })
          cb.assign(i, i + 1)
        })
        m2 - m
      }
      PCode(rt, code)
    }

    registerIEmitCode1("dosage", TArray(tv("N", "float64")), TFloat64,  (_: Type, _: PType) => PFloat64()
    ) { case (cb, r, rt, gp) =>
      gp.flatMap(cb) { case (gpc: PIndexableCode) =>
        val gpv = gpc.memoize(cb, "dosage_gp")

        cb.ifx(gpv.loadLength().cne(3),
          cb._fatal(const("length of gp array must be 3, got ").concat(gpv.loadLength().toS)))

        gpv.loadElement(cb, 1).flatMap(cb) { (_1: PCode) =>
          gpv.loadElement(cb, 2).map(cb) { (_2: PCode) =>
            PCode(rt, _1.tcode[Double] + _2.tcode[Double] * 2.0)
          }
        }
      }
    }
  }
}
