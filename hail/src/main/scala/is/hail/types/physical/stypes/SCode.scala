package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._

object SCode {
  def add(cb: EmitCodeBuilder, left: SCode, right: SCode, required: Boolean): SCode = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Code(left.asInt.intCode(cb) + right.asInt.intCode(cb))
      case (SFloat32, SFloat32) => new SFloat32Code(left.asFloat.floatCode(cb) + right.asFloat.floatCode(cb))
      case (SInt64, SInt64) => new SInt64Code(left.asLong.longCode(cb) + right.asLong.longCode(cb))
      case (SFloat64, SFloat64) => new SFloat64Code(left.asDouble.doubleCode(cb) + right.asDouble.doubleCode(cb))
    }
  }

  def multiply(cb: EmitCodeBuilder, left: SCode, right: SCode, required: Boolean): SCode = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Code(left.asInt.intCode(cb) * right.asInt.intCode(cb))
      case (SFloat32, SFloat32) => new SFloat32Code(left.asFloat.floatCode(cb) * right.asFloat.floatCode(cb))
      case (SInt64, SInt64) => new SInt64Code(left.asLong.longCode(cb) * right.asLong.longCode(cb))
      case (SFloat64, SFloat64) => new SFloat64Code(left.asDouble.doubleCode(cb) * right.asDouble.doubleCode(cb))
    }
  }

  def subtract(cb: EmitCodeBuilder, left: SCode, right: SCode, required: Boolean): SCode = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Code(left.asInt.intCode(cb) - right.asInt.intCode(cb))
      case (SFloat32, SFloat32) => new SFloat32Code(left.asFloat.floatCode(cb) - right.asFloat.floatCode(cb))
      case (SInt64, SInt64) => new SInt64Code(left.asLong.longCode(cb) - right.asLong.longCode(cb))
      case (SFloat64, SFloat64) => new SFloat64Code(left.asDouble.doubleCode(cb) - right.asDouble.doubleCode(cb))
    }
  }

  def divide(cb: EmitCodeBuilder, left: SCode, right: SCode, required: Boolean): SCode = {
    (left.st, right.st) match {
      case (SInt32, SInt32) => new SInt32Code(left.asInt.intCode(cb) / right.asInt.intCode(cb))
      case (SFloat32, SFloat32) => new SFloat32Code(left.asFloat.floatCode(cb) / right.asFloat.floatCode(cb))
      case (SInt64, SInt64) => new SInt64Code(left.asLong.longCode(cb) / right.asLong.longCode(cb))
      case (SFloat64, SFloat64) => new SFloat64Code(left.asDouble.doubleCode(cb) / right.asDouble.doubleCode(cb))
    }
  }

  def _empty: SCode = SVoidCode
}

abstract class SCode {

  def st: SType

  def codeTuple(): IndexedSeq[Code[_]]

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

  def asShuffle: SShuffleCode = asInstanceOf[SShuffleCode]

  def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: SType): SCode =
    castTo(cb, region, destType, false)

  def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: SType, deepCopy: Boolean): SCode = {
    destType.coerceOrCopy(cb, region, this, deepCopy)
  }
  def copyToRegion(cb: EmitCodeBuilder, region: Value[Region], destType: SType): SCode =
    destType.coerceOrCopy(cb, region, this, deepCopy = true)

  def memoize(cb: EmitCodeBuilder, name: String): SValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SValue
}

trait SValue {
  def st: SType

  def get: SCode
}


trait SSettable extends SValue {
  def store(cb: EmitCodeBuilder, v: SCode): Unit

  def settableTuple(): IndexedSeq[Settable[_]]

  def load(): SCode = get
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

  def code: Code[_] = unsupported

  def codeTuple(): IndexedSeq[Code[_]] = unsupported

  def memoizeField(cb: EmitCodeBuilder, name: String): SValue = unsupported
}