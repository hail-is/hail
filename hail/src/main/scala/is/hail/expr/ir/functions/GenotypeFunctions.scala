package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.types.{coerce => _, _}
import is.hail.expr.ir._
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.physical.stypes.primitives.{SFloat64, SInt32}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.{PArray, PCode, PFloat64, PIndexableCode, PInt32, PType}
import is.hail.types.virtual.{TArray, TFloat64, TInt32, Type}

object GenotypeFunctions extends RegistryFunctions {

  def registerAll() {
    registerPCode1("gqFromPL", TArray(tv("N", "int32")), TInt32, (_: Type, _: SType) => SInt32)
    { case (r, cb, rt, _pl: PIndexableCode) =>
      val code = EmitCodeBuilder.scopedCode(r.mb) { cb =>
        val pl = _pl.memoize(cb, "plv")
        val m = cb.newLocal[Int]("m", 99)
        val m2 = cb.newLocal[Int]("m2", 99)
        val i = cb.newLocal[Int]("i", 0)

        cb.whileLoop(i < pl.loadLength(), {
          val value = pl.loadElement(cb, i).get(cb, "PL cannot have missing elements.")
          val pli = cb.newLocal[Int]("pli", value.asInt.intCode(cb))
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
      primitive(code)
    }

    registerIEmitCode1("dosage", TArray(tv("N", "float64")), TFloat64,
      (_: Type, arrayType: EmitType) => EmitType(SFloat64, arrayType.required && arrayType.st.asInstanceOf[SContainer].elementEmitType.required)
    ) { case (cb, r, rt, gp) =>
      gp.toI(cb).flatMap(cb) { case (gpc: PIndexableCode) =>
        val gpv = gpc.memoize(cb, "dosage_gp")

        cb.ifx(gpv.loadLength().cne(3),
          cb._fatal(const("length of gp array must be 3, got ").concat(gpv.loadLength().toS)))

        gpv.loadElement(cb, 1).flatMap(cb) { (_1: SCode) =>
          gpv.loadElement(cb, 2).map(cb) { (_2: SCode) =>
            primitive(_1.asDouble.doubleCode(cb) + _2.asDouble.doubleCode(cb) * 2.0)
          }
        }
      }
    }
  }
}
