package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.concrete.SRNGStateValue
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._

object SCode {
  def add(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.value + right.asInt.value))
      case (SFloat32, SFloat32) =>
        new SFloat32Value(cb.memoize(left.asFloat.value + right.asFloat.value))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.value + right.asLong.value))
      case (SFloat64, SFloat64) =>
        new SFloat64Value(cb.memoize(left.asDouble.value + right.asDouble.value))
    }
  }

  def multiply(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.value * right.asInt.value))
      case (SFloat32, SFloat32) =>
        new SFloat32Value(cb.memoize(left.asFloat.value * right.asFloat.value))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.value * right.asLong.value))
      case (SFloat64, SFloat64) =>
        new SFloat64Value(cb.memoize(left.asDouble.value * right.asDouble.value))
    }
  }

  def subtract(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.value - right.asInt.value))
      case (SFloat32, SFloat32) =>
        new SFloat32Value(cb.memoize(left.asFloat.value - right.asFloat.value))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.value - right.asLong.value))
      case (SFloat64, SFloat64) =>
        new SFloat64Value(cb.memoize(left.asDouble.value - right.asDouble.value))
    }
  }

  def divide(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.value / right.asInt.value))
      case (SFloat32, SFloat32) =>
        new SFloat32Value(cb.memoize(left.asFloat.value / right.asFloat.value))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.value / right.asLong.value))
      case (SFloat64, SFloat64) =>
        new SFloat64Value(cb.memoize(left.asDouble.value / right.asDouble.value))
    }
  }

  def _empty: SValue = SVoidValue
}

trait SValue {
  def st: SType

  def valueTuple: IndexedSeq[Value[_]]

  def asBoolean: SBooleanValue = asInstanceOf[SBooleanValue]

  def asInt: SInt32Value = asInstanceOf[SInt32Value]

  def asInt32: SInt32Value = asInstanceOf[SInt32Value]

  def asLong: SInt64Value = asInstanceOf[SInt64Value]

  def asInt64: SInt64Value = asInstanceOf[SInt64Value]

  def asFloat: SFloat32Value = asInstanceOf[SFloat32Value]

  def asFloat32: SFloat32Value = asInstanceOf[SFloat32Value]

  def asFloat64: SFloat64Value = asInstanceOf[SFloat64Value]

  def asDouble: SFloat64Value = asInstanceOf[SFloat64Value]

  def asPrimitive: SPrimitiveValue = asInstanceOf[SPrimitiveValue]

  def asBinary: SBinaryValue = asInstanceOf[SBinaryValue]

  def asIndexable: SIndexableValue = asInstanceOf[SIndexableValue]

  def asBaseStruct: SBaseStructValue = asInstanceOf[SBaseStructValue]

  def asString: SStringValue = asInstanceOf[SStringValue]

  def asInterval: SIntervalValue = asInstanceOf[SIntervalValue]

  def asNDArray: SNDArrayValue = asInstanceOf[SNDArrayValue]

  def asLocus: SLocusValue = asInstanceOf[SLocusValue]

  def asCall: SCallValue = asInstanceOf[SCallValue]

  def asStream: SStreamValue = asInstanceOf[SStreamValue]

  def asRNGState: SRNGStateValue = asInstanceOf[SRNGStateValue]

  def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: SType): SValue =
    castTo(cb, region, destType, false)

  def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: SType, deepCopy: Boolean)
    : SValue =
    destType.coerceOrCopy(cb, region, this, deepCopy)

  def copyToRegion(cb: EmitCodeBuilder, region: Value[Region], destType: SType): SValue =
    destType.coerceOrCopy(cb, region, this, deepCopy = true)

  def hash(cb: EmitCodeBuilder): SInt32Value =
    throw new UnsupportedOperationException(s"Stype $st has no hashcode")

  def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value
}

trait SSettable extends SValue {
  def store(cb: EmitCodeBuilder, v: SValue): Unit

  def settableTuple(): IndexedSeq[Settable[_]]
}

object SSettable {
  def apply(sb: SettableBuilder, st: SType, name: String): SSettable =
    st.fromSettables(st.settableTupleTypes().zipWithIndex.map { case (ti, i) =>
      sb.newSettable(s"${name}_${st.getClass.getSimpleName}_$i")(ti)
    })
}

trait SUnrealizableValue extends SValue {
  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value =
    throw new UnsupportedOperationException(s"Unrealizable SValue has no size in bytes.")
}
