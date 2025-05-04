package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._

// Consider a cast from t1 to t2 to be lossless when casting from t1 to t2 and back is identity.
case class CastInfo(impl: (EmitCodeBuilder, SValue) => SValue, isLossless: Boolean = false) {
  def apply(cb: EmitCodeBuilder, x: SValue): SValue = impl(cb, x)
}

object Casts {
  private val casts: Map[(Type, Type), CastInfo] = Map(
    (TInt32, TInt32) -> CastInfo((_, x) => x, isLossless = true),
    (TInt32, TInt64) -> CastInfo(
      (cb, x) => primitive(cb.memoize(x.asInt.value.toL)),
      isLossless = true,
    ),
    (TInt32, TFloat32) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asInt.value.toF))
    ),
    (TInt32, TFloat64) -> CastInfo(
      (cb, x) => primitive(cb.memoize(x.asInt.value.toD)),
      isLossless = true,
    ),
    (TInt64, TInt32) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asLong.value.toI))
    ),
    (TInt64, TInt64) -> CastInfo((_, x) => x, isLossless = true),
    (TInt64, TFloat32) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asLong.value.toF))
    ),
    (TInt64, TFloat64) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asLong.value.toD))
    ),
    (TFloat32, TInt32) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asFloat.value.toI))
    ),
    (TFloat32, TInt64) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asFloat.value.toL))
    ),
    (TFloat32, TFloat32) -> CastInfo((_, x) => x, isLossless = true),
    (TFloat32, TFloat64) -> CastInfo(
      (cb, x) => primitive(cb.memoize(x.asFloat.value.toD)),
      isLossless = true,
    ),
    (TFloat64, TInt32) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asDouble.value.toI))
    ),
    (TFloat64, TInt64) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asDouble.value.toL))
    ),
    (TFloat64, TFloat32) -> CastInfo((cb, x) =>
      primitive(cb.memoize(x.asDouble.value.toF))
    ),
    (TFloat64, TFloat64) -> CastInfo((_, x) => x, isLossless = true),
  )

  def get(from: Type, to: Type): CastInfo = casts(from -> to)

  def valid(from: Type, to: Type): Boolean =
    casts.contains(from -> to)
}
