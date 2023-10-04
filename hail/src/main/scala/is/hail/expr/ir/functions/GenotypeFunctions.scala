package is.hail.expr.ir.functions

import is.hail.asm4s.{coerce => _, _}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SFloat64, SInt32}
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.virtual.{TArray, TFloat64, TInt32, Type}
import is.hail.types.{tcoerce => _}

object GenotypeFunctions extends RegistryFunctions {

  def registerAll() {
    registerSCode1("gqFromPL", TArray(tv("N", "int32")), TInt32, (_: Type, _: SType) => SInt32)
    { case (r, cb, rt, pl: SIndexableValue, errorID) =>
      val m = cb.newLocal[Int]("m", 99)
      val m2 = cb.newLocal[Int]("m2", 99)
      val i = cb.newLocal[Int]("i", 0)

      cb.while_(i < pl.loadLength(), {
        val value = pl.loadElement(cb, i).get(cb, "PL cannot have missing elements.", errorID)
        val pli = cb.newLocal[Int]("pli", value.asInt.value)
        cb.ifx(pli < m, {
          cb.assign(m2, m)
          cb.assign(m, pli)
        }, {
          cb.ifx(pli < m2,
            cb.assign(m2, pli))
        })
        cb.assign(i, i + 1)
      })

      primitive(cb.memoize(m2 - m))
    }

    registerIEmitCode1("dosage", TArray(tv("N", "float64")), TFloat64,
      (_: Type, arrayType: EmitType) => EmitType(SFloat64, arrayType.required && arrayType.st.asInstanceOf[SContainer].elementEmitType.required)
    ) { case (cb, r, rt, errorID, gp) =>
      gp.toI(cb).flatMap(cb) { case gpv: SIndexableValue =>
        cb.ifx(gpv.loadLength().cne(3),
          cb._fatalWithError(errorID, const("length of gp array must be 3, got ").concat(gpv.loadLength().toS)))

        gpv.loadElement(cb, 1).flatMap(cb) { _1 =>
          gpv.loadElement(cb, 2).map(cb) { _2 =>
            primitive(cb.memoize(_1.asDouble.value + _2.asDouble.value * 2.0))
          }
        }
      }
    }
  }
}
