package is.hail.expr.ir.functions

import is.hail.asm4s.{coerce => _, _}
import is.hail.types.{tcoerce => _}
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SFloat64, SInt32, SInt32Value}
import is.hail.types.virtual.{TArray, TFloat64, TInt32, Type}

object GenotypeFunctions extends RegistryFunctions {

  override def registerAll(): Unit = {
    registerSCode1("gqFromPL", TArray(tv("N", "int32")), TInt32, (_: Type, _: SType) => SInt32) {
      case (_, cb, _, pl: SIndexableValue, errorID @ _) =>
        val m = cb.newLocal[Int]("m", 99)
        val m2 = cb.newLocal[Int]("m2", 99)

        pl.forEachDefined(cb) { case (cb, _, pli: SInt32Value) =>
          cb.if_(
            pli.value < m, {
              cb.assign(m2, m)
              cb.assign(m, pli.value)
            },
            cb.if_(pli.value < m2, cb.assign(m2, pli.value)),
          )
        }

        primitive(cb.memoize(m2 - m))
    }

    registerIEmitCode1(
      "dosage",
      TArray(tv("N", "float64")),
      TFloat64,
      (_: Type, arrayType: EmitType) =>
        EmitType(
          SFloat64,
          arrayType.required && arrayType.st.asInstanceOf[SContainer].elementEmitType.required,
        ),
    ) { case (cb, _, _, errorID, gp) =>
      gp.toI(cb).flatMap(cb) { case gpv: SIndexableValue =>
        cb.if_(
          gpv.loadLength.cne(3),
          cb._fatalWithError(
            errorID,
            const("length of gp array must be 3, got ").concat(gpv.loadLength.toS),
          ),
        )

        gpv.loadElement(cb, 1).flatMap(cb) { _1 =>
          gpv.loadElement(cb, 2).map(cb) { _2 =>
            primitive(cb.memoize(_1.asDouble.value + _2.asDouble.value * 2.0))
          }
        }
      }
    }
  }
}
