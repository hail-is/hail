package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.streams.{StreamArgType, StreamProducer}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.interfaces.{SStream, SStreamValue}
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils._

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

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue

  def virtualType: Type

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode

  def loadedSType: SType
}

case object Int32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = IntInfo

  override def loadedSType: SType = SInt32

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue = new SInt32Value(coerce[Int](c))

  def virtualType: Type = TInt32

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode =
    SingleCodeSCode(this, pc.asInt.value)
}

case object Int64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = SInt64

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue = new SInt64Value(coerce[Long](c))

  def virtualType: Type = TInt64

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode =
    SingleCodeSCode(this, pc.asLong.value)
}

case object Float32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = FloatInfo

  override def loadedSType: SType = SFloat32

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue = new SFloat32Value(coerce[Float](c))

  def virtualType: Type = TFloat32

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode =
    SingleCodeSCode(this, pc.asFloat.value)
}

case object Float64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = DoubleInfo

  override def loadedSType: SType = SFloat64

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue = new SFloat64Value(coerce[Double](c))

  def virtualType: Type = TFloat64

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode =
    SingleCodeSCode(this, pc.asDouble.value)
}

case object BooleanSingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = BooleanInfo

  override def loadedSType: SType = SBoolean

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue = new SBooleanValue(coerce[Boolean](c))

  def virtualType: Type = TBoolean

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode =
    SingleCodeSCode(this, pc.asBoolean.value)
}

case class StreamSingleCodeType(requiresMemoryManagementPerElement: Boolean, eltType: PType) extends SingleCodeType {
  self =>

  override def loadedSType: SType = SStream(EmitType(eltType.sType, true))

  def virtualType: Type = TStream(eltType.virtualType)

  def ti: TypeInfo[_] = classInfo[StreamArgType]

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue = {
    val mb = cb.emb
    val xIter = mb.genFieldThisRef[Iterator[java.lang.Long]]("streamInIterator")

    // this, Region, ...
    val mkIter = coerce[StreamArgType](c)
    val eltRegion = mb.genFieldThisRef[Region]("stream_input_element_region")
    val rvAddr = mb.genFieldThisRef[Long]("stream_input_addr")

    val producer = new StreamProducer {
      override val length: Option[EmitCodeBuilder => Code[Int]] = None

      override def initialize(cb: EmitCodeBuilder): Unit = {
        cb.assign(xIter, mkIter.invoke[Region, Region, Iterator[java.lang.Long]]("apply", r, eltRegion))
      }

      override val elementRegion: Settable[Region] = eltRegion
      override val requiresMemoryManagementPerElement: Boolean = self.requiresMemoryManagementPerElement
      override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
        // NB: locals should not be used in this implementation. The way In() nodes are
        // stored in fields at the beginning of code generation leads to the method builder
        // here being different from the method the stream will eventually be consumed in
        cb.ifx(!xIter.load().hasNext, cb.goto(LendOfStream))
        cb.assign(rvAddr, xIter.load().next().invoke[Long]("longValue"))
        cb.goto(LproduceElementDone)
      }

      override val element: EmitCode =
        EmitCode.fromI(mb)(cb => IEmitCode.present(cb, cb.memoizeField(eltType.loadCheapSCode(cb, rvAddr), "stream_input_element")))

      override def close(cb: EmitCodeBuilder): Unit = {}
    }
    SStreamValue(SStream(EmitType(eltType.sType, true)), producer)
  }

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode =
    throw new UnsupportedOperationException
}

case class PTypeReferenceSingleCodeType(pt: PType) extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = pt.sType

  def loadToSValue(cb: EmitCodeBuilder, r: Value[Region], c: Value[_]): SValue =
    cb.memoizeField(pt.loadCheapSCode(cb, coerce[Long](c)), "PTypeReferenceSingleCodeType")

  def virtualType: Type = pt.virtualType

  def coerceSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean): SingleCodeSCode = {
    SingleCodeSCode(this, pt.store(cb, region, pc, deepCopy = deepCopy))
  }
}

object SingleCodeSCode {
  def fromSCode(cb: EmitCodeBuilder, pc: SValue, region: Value[Region], deepCopy: Boolean = false): SingleCodeSCode = {
    SingleCodeType.fromSType(pc.st).coerceSCode(cb, pc, region, deepCopy)
  }
}

case class SingleCodeSCode(typ: SingleCodeType, code: Value[_])