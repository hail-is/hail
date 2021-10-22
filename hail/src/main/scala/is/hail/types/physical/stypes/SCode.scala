package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._

object SCode {
  def add(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.intCode(cb) + right.asInt.intCode(cb)))
      case (SFloat32, SFloat32) => new SFloat32Value(cb.memoize(left.asFloat.floatCode(cb) + right.asFloat.floatCode(cb)))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.longCode(cb) + right.asLong.longCode(cb)))
      case (SFloat64, SFloat64) => new SFloat64Value(cb.memoize(left.asDouble.doubleCode(cb) + right.asDouble.doubleCode(cb)))
    }
  }

  def multiply(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.intCode(cb) * right.asInt.intCode(cb)))
      case (SFloat32, SFloat32) => new SFloat32Value(cb.memoize(left.asFloat.floatCode(cb) * right.asFloat.floatCode(cb)))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.longCode(cb) * right.asLong.longCode(cb)))
      case (SFloat64, SFloat64) => new SFloat64Value(cb.memoize(left.asDouble.doubleCode(cb) * right.asDouble.doubleCode(cb)))
    }
  }

  def subtract(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.intCode(cb) - right.asInt.intCode(cb)))
      case (SFloat32, SFloat32) => new SFloat32Value(cb.memoize(left.asFloat.floatCode(cb) - right.asFloat.floatCode(cb)))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.longCode(cb) - right.asLong.longCode(cb)))
      case (SFloat64, SFloat64) => new SFloat64Value(cb.memoize(left.asDouble.doubleCode(cb) - right.asDouble.doubleCode(cb)))
    }
  }

  def divide(cb: EmitCodeBuilder, left: SValue, right: SValue, required: Boolean): SValue = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Value(cb.memoize(left.asInt.intCode(cb) / right.asInt.intCode(cb)))
      case (SFloat32, SFloat32) => new SFloat32Value(cb.memoize(left.asFloat.floatCode(cb) / right.asFloat.floatCode(cb)))
      case (SInt64, SInt64) => new SInt64Value(cb.memoize(left.asLong.longCode(cb) / right.asLong.longCode(cb)))
      case (SFloat64, SFloat64) => new SFloat64Value(cb.memoize(left.asDouble.doubleCode(cb) / right.asDouble.doubleCode(cb)))
    }
  }

  def _empty: SValue = SVoidValue
}

abstract class SCode {

  def st: SType

  def asBoolean: SBooleanCode = asInstanceOf[SBooleanCode]

  def asInt: SInt32Code = asInstanceOf[SInt32Code]

  def asInt32: SInt32Code = asInstanceOf[SInt32Code]

  def asLong: SInt64Code = asInstanceOf[SInt64Code]

  def asInt64: SInt64Code = asInstanceOf[SInt64Code]

  def asFloat: SFloat32Code = asInstanceOf[SFloat32Code]

  def asFloat32: SFloat32Code = asInstanceOf[SFloat32Code]

  def asFloat64: SFloat64Code = asInstanceOf[SFloat64Code]

  def asDouble: SFloat64Code = asInstanceOf[SFloat64Code]

  def asPrimitive: SPrimitiveCode = asInstanceOf[SPrimitiveCode]

  def asBinary: SBinaryCode = asInstanceOf[SBinaryCode]

  def asIndexable: SIndexableCode = asInstanceOf[SIndexableCode]

  def asBaseStruct: SBaseStructCode = asInstanceOf[SBaseStructCode]

  def asString: SStringCode = asInstanceOf[SStringCode]

  def asInterval: SIntervalCode = asInstanceOf[SIntervalCode]

  def asNDArray: SNDArrayCode = asInstanceOf[SNDArrayCode]

  def asLocus: SLocusCode = asInstanceOf[SLocusCode]

  def asCall: SCallCode = asInstanceOf[SCallCode]

  def asStream: SStreamCode = asInstanceOf[SStreamCode]
}

trait SValue {
  def st: SType

  def valueTuple: IndexedSeq[Value[_]]

  def get: SCode

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

  def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: SType): SValue =
    castTo(cb, region, destType, false)

  def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: SType, deepCopy: Boolean): SValue = {
    destType.coerceOrCopy(cb, region, this, deepCopy)
  }

  def copyToRegion(cb: EmitCodeBuilder, region: Value[Region], destType: SType): SValue =
    destType.coerceOrCopy(cb, region, this, deepCopy = true)

  def hash(cb: EmitCodeBuilder): SInt32Value = throw new UnsupportedOperationException(s"Stype ${st} has no hashcode")
}


trait SSettable extends SValue {
  def store(cb: EmitCodeBuilder, v: SCode): Unit

  def store(cb: EmitCodeBuilder, v: SValue): Unit =
    store(cb, v.get)

  def settableTuple(): IndexedSeq[Settable[_]]
}

object SSettable {
  def apply(sb: SettableBuilder, st: SType, name: String): SSettable = {
    st.fromSettables(st.settableTupleTypes().zipWithIndex.map { case (ti, i) =>
      sb.newSettable(s"${ name }_${ st.getClass.getSimpleName }_$i")(ti)
    })
  }
}

trait SUnrealizableCode extends SCode {
  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$this is not realizable")
}

trait SUnrealizableValue extends SValue