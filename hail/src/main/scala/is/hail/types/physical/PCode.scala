package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.streams.{StreamArgType, StreamProducer}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{PVoidCode, SStream, SStreamCode}
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils._

trait PValue extends SValue {
  pValueSelf =>
  def get: PCode
}

trait PSettable extends PValue with SSettable {

  def store(cb: EmitCodeBuilder, v: SCode): Unit = store(cb, v.asInstanceOf[PCode])

  def store(cb: EmitCodeBuilder, v: PCode): Unit

  def settableTuple(): IndexedSeq[Settable[_]]

  override def load(): PCode = get
}

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

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode

  def virtualType: Type

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode

  def loadedSType: SType
}

case object Int32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = IntInfo

  override def loadedSType: SType = SInt32

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SInt32Code(coerce[Int](c))

  def virtualType: Type = TInt32

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asInt.intCode(cb))
}

case object Int64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = SInt64

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SInt64Code(coerce[Long](c))

  def virtualType: Type = TInt64

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asLong.longCode(cb))
}

case object Float32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = FloatInfo

  override def loadedSType: SType = SFloat32

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SFloat32Code(coerce[Float](c))

  def virtualType: Type = TFloat32

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asFloat.floatCode(cb))
}

case object Float64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = DoubleInfo

  override def loadedSType: SType = SFloat64

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SFloat64Code(coerce[Double](c))

  def virtualType: Type = TFloat64

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asDouble.doubleCode(cb))
}

case object BooleanSingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = BooleanInfo

  override def loadedSType: SType = SBoolean

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SBooleanCode(coerce[Boolean](c))

  def virtualType: Type = TBoolean

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = SingleCodePCode(this, pc.asBoolean.boolCode(cb))
}

case class StreamSingleCodeType(requiresMemoryManagementPerElement: Boolean, eltType: PType) extends SingleCodeType {
  self =>

  override def loadedSType: SType = SStream(EmitType(eltType.sType, true))

  def virtualType: Type = TStream(eltType.virtualType)

  def ti: TypeInfo[_] = classInfo[StreamArgType]

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = {
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

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = throw new UnsupportedOperationException
}

case class PTypeReferenceSingleCodeType(pt: PType) extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  override def loadedSType: SType = pt.sType

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = pt.loadCheapPCode(cb, coerce[Long](c))

  def virtualType: Type = pt.virtualType

  def coercePCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean): SingleCodePCode = {
    SingleCodePCode(this, pt.store(cb, region, pc, deepCopy = deepCopy))
  }
}

object SingleCodePCode {
  def fromPCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region], deepCopy: Boolean = false): SingleCodePCode = {
    SingleCodeType.fromSType(pc.st).coercePCode(cb, pc, region, deepCopy)
  }
}

case class SingleCodePCode(typ: SingleCodeType, code: Code[_])

abstract class PCode extends SCode {
  self =>

  def st: SType

  def code: Code[_]

  def codeTuple(): IndexedSeq[Code[_]]

  override def asBoolean: SBooleanCode = asInstanceOf[SBooleanCode]

  override def asInt: SInt32Code = asInstanceOf[SInt32Code]

  override def asInt32: SInt32Code = asInstanceOf[SInt32Code]

  override def asLong: SInt64Code = asInstanceOf[SInt64Code]

  override def asInt64: SInt64Code = asInstanceOf[SInt64Code]

  override def asFloat: SFloat32Code = asInstanceOf[SFloat32Code]

  override def asFloat32: SFloat32Code = asInstanceOf[SFloat32Code]

  override def asFloat64: SFloat64Code = asInstanceOf[SFloat64Code]

  override def asDouble: SFloat64Code = asInstanceOf[SFloat64Code]

  override def asBinary: PBinaryCode = asInstanceOf[PBinaryCode]

  override def asIndexable: PIndexableCode = asInstanceOf[PIndexableCode]

  override def asBaseStruct: PBaseStructCode = asInstanceOf[PBaseStructCode]

  override def asString: PStringCode = asInstanceOf[PStringCode]

  override def asInterval: PIntervalCode = asInstanceOf[PIntervalCode]

  override def asNDArray: PNDArrayCode = asInstanceOf[PNDArrayCode]

  override def asLocus: PLocusCode = asInstanceOf[PLocusCode]

  override def asCall: PCallCode = asInstanceOf[PCallCode]

  def memoize(cb: EmitCodeBuilder, name: String): PValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue

  final def toPCode(cb: EmitCodeBuilder, region: Value[Region]): PCode = this
}

object PCode {
//  def apply(pt: PType, code: Code[_]): PCode = pt.setRequired(false) match {
//    case pt: PCanonicalArray =>
//      new SIndexablePointerCode(SIndexablePointer(pt), coerce[Long](code))
//    case pt: PCanonicalSet =>
//      new SIndexablePointerCode(SIndexablePointer(pt), coerce[Long](code))
//    case pt: PCanonicalDict =>
//      new SIndexablePointerCode(SIndexablePointer(pt), coerce[Long](code))
//    case pt: PCanonicalBaseStruct =>
//      new SBaseStructPointerCode(SBaseStructPointer(pt), coerce[Long](code))
//    case pt: PCanonicalBinary =>
//      new SBinaryPointerCode(SBinaryPointer(pt), coerce[Long](code))
//    case pt: PCanonicalShuffle =>
//      new SCanonicalShufflePointerCode(SCanonicalShufflePointer(pt),
//        new SBinaryPointerCode(SBinaryPointer(pt.representation), coerce[Long](code)))
//    case pt: PCanonicalString =>
//      new SStringPointerCode(SStringPointer(pt), coerce[Long](code))
//    case pt: PCanonicalInterval =>
//      new SIntervalPointerCode(SIntervalPointer(pt), coerce[Long](code))
//    case pt: PCanonicalLocus =>
//      new SCanonicalLocusPointerCode(SCanonicalLocusPointer(pt), coerce[Long](code))
//    case pt: PCanonicalCall =>
//      new SCanonicalCallCode(coerce[Int](code))
//    case pt: PCanonicalNDArray =>
//      new SNDArrayPointerCode(SNDArrayPointer(pt), coerce[Long](code))
//    case pt: PCanonicalStream =>
//      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
//    case PVoid =>
//      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
//    case PBoolean(r) =>
//      new SBooleanCode(coerce[Boolean](code))
//    case PInt32(r) =>
//      new SInt32Code(coerce[Int](code))
//    case PInt64(r) =>
//      new SInt64Code(coerce[Long](code))
//    case PFloat32(r) =>
//      new SFloat32Code(coerce[Float](code))
//    case PFloat64(r) =>
//      new SFloat64Code(coerce[Double](code))
//  }

  def _empty: PCode = PVoidCode
}

object PSettable {
  def apply(sb: SettableBuilder, st: SType, name: String): PSettable = {
    st.fromSettables(st.settableTupleTypes().zipWithIndex.map { case (ti, i) =>
      sb.newSettable(s"${ name }_${ st.getClass.getSimpleName }_$i")(ti)
    }).asInstanceOf[PSettable]
  }
}
