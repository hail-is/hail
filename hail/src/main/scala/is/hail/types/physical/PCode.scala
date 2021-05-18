package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.streams.{StreamArgType, StreamProducer}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces.{SStream, SStreamCode}
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
    case _ => PTypeReferenceSingleCodeType(t.canonicalPType().setRequired(true))

  }
}

sealed trait SingleCodeType {
  def ti: TypeInfo[_]

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode

  def virtualType: Type

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode

  def loadedSType: SType
}

case object Int32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = IntInfo

  override def loadedSType: SType = SInt32

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode = new SInt32Code(coerce[Int](c))

  def virtualType: Type = TInt32

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asInt.intCode(cb))
}

case object Int64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = SInt64

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode = new SInt64Code(coerce[Long](c))

  def virtualType: Type = TInt64

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asLong.longCode(cb))
}

case object Float32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = FloatInfo

  override def loadedSType: SType = SFloat32

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode = new SFloat32Code(coerce[Float](c))

  def virtualType: Type = TFloat32

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asFloat.floatCode(cb))
}

case object Float64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = DoubleInfo

  override def loadedSType: SType = SFloat64

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode = new SFloat64Code(coerce[Double](c))

  def virtualType: Type = TFloat64

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asDouble.doubleCode(cb))
}

case object BooleanSingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = BooleanInfo

  override def loadedSType: SType = SBoolean

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode = new SBooleanCode(coerce[Boolean](c))

  def virtualType: Type = TBoolean

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asBoolean.boolCode(cb))
}

case class StreamSingleCodeType(requiresMemoryManagementPerElement: Boolean, eltType: PType) extends SingleCodeType {
  self =>

  override def loadedSType: SType = SStream(EmitType(eltType.sType, true))

  def virtualType: Type = TStream(eltType.virtualType)

  def ti: TypeInfo[_] = classInfo[StreamArgType]

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode = {
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
        val hasNext = cb.newLocal[Boolean]("stream_in_hasnext", xIter.load().hasNext)
        cb.ifx(!hasNext, cb.goto(LendOfStream))
        cb.assign(rvAddr, xIter.load().next().invoke[Long]("longValue"))
        cb.goto(LproduceElementDone)
      }

      override val element: EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, eltType.loadCheapPCode(cb, rvAddr)))

      override def close(cb: EmitCodeBuilder): Unit = {}
    }
    SStreamCode(SStream(EmitType(eltType.sType, true)), producer)
  }

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = throw new UnsupportedOperationException
}

case class PTypeReferenceSingleCodeType(pt: PType) extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = pt.sType

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): SCode = pt.loadCheapPCode(cb, coerce[Long](c))

  def virtualType: Type = pt.virtualType

  def coercePCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = {
    SingleCodePCode(this, pt.store(cb, region, pc, deepCopy = deepCopy))
  }
}

object SingleCodePCode {
  def fromPCode(cb: EmitCodeBuilder, pc: SCode, region: Value[Region], deepCopy: Boolean = false): SingleCodePCode = {
    SingleCodeType.fromSType(pc.st).coercePCode(cb, pc, region, deepCopy)
  }
}

case class SingleCodePCode(typ: SingleCodeType, code: Code[_])
