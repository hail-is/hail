package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.types._
import is.hail.types.physical.stypes.{SCode, SValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._

import scala.language.existentials

object Casts {
  private val casts: Map[(Type, Type), (EmitCodeBuilder, SValue) => SValue] = Map(
    (TInt32, TInt32) -> ((cb: EmitCodeBuilder, x: SValue) => x),
    (TInt32, TInt64) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asInt.intCode(cb).toL))),
    (TInt32, TFloat32) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asInt.intCode(cb).toF))),
    (TInt32, TFloat64) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asInt.intCode(cb).toD))),
    (TInt64, TInt32) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asLong.longCode(cb).toI))),
    (TInt64, TInt64) -> ((cb: EmitCodeBuilder, x: SValue) => x),
    (TInt64, TFloat32) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asLong.longCode(cb).toF))),
    (TInt64, TFloat64) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asLong.longCode(cb).toD))),
    (TFloat32, TInt32) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asFloat.floatCode(cb).toI))),
    (TFloat32, TInt64) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asFloat.floatCode(cb).toL))),
    (TFloat32, TFloat32) -> ((cb: EmitCodeBuilder, x: SValue) => x),
    (TFloat32, TFloat64) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asFloat.floatCode(cb).toD))),
    (TFloat64, TInt32) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asDouble.doubleCode(cb).toI))),
    (TFloat64, TInt64) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asDouble.doubleCode(cb).toL))),
    (TFloat64, TFloat32) -> ((cb: EmitCodeBuilder, x: SValue) => primitive(cb.memoize(x.asDouble.doubleCode(cb).toF))),
    (TFloat64, TFloat64) -> ((cb: EmitCodeBuilder, x: SValue) => x))

  def get(from: Type, to: Type): (EmitCodeBuilder, SValue) => SValue =
    casts(from -> to)

  def valid(from: Type, to: Type): Boolean =
    casts.contains(from -> to)
}
