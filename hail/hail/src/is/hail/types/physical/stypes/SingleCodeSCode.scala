package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.interfaces.{
  NoBoxLongIterator, SStream, SStreamConcrete, SStreamIteratorLong,
}
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._

object SingleCodeType {
  def typeInfoFromType(t: Type): TypeInfo[_] = t match {
    case TInt32 => IntInfo
    case TInt64 => LongInfo
    case TFloat32 => FloatInfo
    case TFloat64 => DoubleInfo
    case TBoolean => BooleanInfo
    case TVoid => UnitInfo
    case _ => LongInfo // all others passed as ptype references
  }

  def fromSType(t: SType): SingleCodeType = t.virtualType match {
    case TInt32 => Int32SingleCodeType
    case TInt64 => Int64SingleCodeType
    case TFloat32 => Float32SingleCodeType
    case TFloat64 => Float64SingleCodeType
    case TBoolean => BooleanSingleCodeType
    case _ => PTypeReferenceSingleCodeType(t.storageType().setRequired(true))

  }
}

sealed trait SingleCodeType {
  def ti: TypeInfo[_]

  def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue

  def virtualType: Type

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean)
    : SingleCodeSCode

  def loadedSType: SType
}

case object Int32SingleCodeType extends SingleCodeType {
  override def ti: TypeInfo[_] = IntInfo

  override def loadedSType: SType = SInt32

  override def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue =
    new SInt32Value(coerce[Int](c))

  override def virtualType: Type = TInt32

  override def coerceSCode(
    cb: EmitCodeBuilder,
    pc: SValue,
    region: Value[Region],
    deepCopy: Boolean,
  ): SingleCodeSCode =
    SingleCodeSCode(this, pc.asInt.value)
}

case object Int64SingleCodeType extends SingleCodeType {
  override def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = SInt64

  override def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue =
    new SInt64Value(coerce[Long](c))

  override def virtualType: Type = TInt64

  override def coerceSCode(
    cb: EmitCodeBuilder,
    pc: SValue,
    region: Value[Region],
    deepCopy: Boolean,
  ): SingleCodeSCode =
    SingleCodeSCode(this, pc.asLong.value)
}

case object Float32SingleCodeType extends SingleCodeType {
  override def ti: TypeInfo[_] = FloatInfo

  override def loadedSType: SType = SFloat32

  override def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue =
    new SFloat32Value(coerce[Float](c))

  override def virtualType: Type = TFloat32

  override def coerceSCode(
    cb: EmitCodeBuilder,
    pc: SValue,
    region: Value[Region],
    deepCopy: Boolean,
  ): SingleCodeSCode =
    SingleCodeSCode(this, pc.asFloat.value)
}

case object Float64SingleCodeType extends SingleCodeType {
  override def ti: TypeInfo[_] = DoubleInfo

  override def loadedSType: SType = SFloat64

  override def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue =
    new SFloat64Value(coerce[Double](c))

  override def virtualType: Type = TFloat64

  override def coerceSCode(
    cb: EmitCodeBuilder,
    pc: SValue,
    region: Value[Region],
    deepCopy: Boolean,
  ): SingleCodeSCode =
    SingleCodeSCode(this, pc.asDouble.value)
}

case object BooleanSingleCodeType extends SingleCodeType {
  override def ti: TypeInfo[_] = BooleanInfo

  override def loadedSType: SType = SBoolean

  override def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue =
    new SBooleanValue(coerce[Boolean](c))

  override def virtualType: Type = TBoolean

  override def coerceSCode(
    cb: EmitCodeBuilder,
    pc: SValue,
    region: Value[Region],
    deepCopy: Boolean,
  ): SingleCodeSCode =
    SingleCodeSCode(this, pc.asBoolean.value)
}

case class StreamSingleCodeType(
  requiresMemoryManagementPerElement: Boolean,
  eltType: PType,
  eltRequired: Boolean,
) extends SingleCodeType {
  self =>

  override def loadedSType: SType = SStream(EmitType(eltType.sType, true))

  override def virtualType: Type = TStream(eltType.virtualType)

  override def ti: TypeInfo[_] = classInfo[NoBoxLongIterator]

  override def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue =
    new SStreamConcrete(
      SStreamIteratorLong(eltRequired, eltType, requiresMemoryManagementPerElement),
      coerce[NoBoxLongIterator](c),
    )

  override def coerceSCode(
    cb: EmitCodeBuilder,
    pc: SValue,
    region: Value[Region],
    deepCopy: Boolean,
  ): SingleCodeSCode =
    throw new UnsupportedOperationException
}

case class PTypeReferenceSingleCodeType(pt: PType) extends SingleCodeType {
  override def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = pt.sType

  override def loadToSValue(cb: EmitCodeBuilder, c: Value[_]): SValue =
    cb.memoizeField(pt.loadCheapSCode(cb, coerce[Long](c)), "PTypeReferenceSingleCodeType")

  override def virtualType: Type = pt.virtualType

  override def coerceSCode(
    cb: EmitCodeBuilder,
    pc: SValue,
    region: Value[Region],
    deepCopy: Boolean,
  ): SingleCodeSCode =
    SingleCodeSCode(this, pt.store(cb, region, pc, deepCopy = deepCopy))
}

object SingleCodeSCode {
  def fromSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean = false)
    : SingleCodeSCode =
    SingleCodeType.fromSType(pc.st).coerceSCode(cb, pc, region, deepCopy)
}

case class SingleCodeSCode(typ: SingleCodeType, code: Value[_])
