package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.expr.ir.Stream.unfold
import is.hail.expr.ir._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{PVoidCode, SStream}
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual.{TBoolean, TFloat32, TFloat64, TInt32, TInt64, TStream, TVoid, Type}
import is.hail.utils._

trait PValue extends SValue { pValueSelf =>
  def pt: PType

  def get: PCode

  def value: Value[_] = {
    new Value[Any] {
      override def get: Code[Any] = pValueSelf.get.code
    }
  }
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
}

sealed trait SingleCodeType {
  def ti: TypeInfo[_]

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode

  def virtualType: Type
}

case object Int32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = IntInfo

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SInt32Code(true, coerce[Int](c))

  def virtualType: Type = TInt32
}

case object Int64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SInt64Code(true, coerce[Long](c))

  def virtualType: Type = TInt64
}

case object Float32SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = FloatInfo

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SFloat32Code(true, coerce[Float](c))

  def virtualType: Type = TFloat32
}

case object Float64SingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = DoubleInfo

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SFloat64Code(true, coerce[Double](c))

  def virtualType: Type = TFloat64
}

case object BooleanSingleCodeType extends SingleCodeType {
  def ti: TypeInfo[_] = BooleanInfo

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = new SBooleanCode(true, coerce[Boolean](c))

  def virtualType: Type = TBoolean
}

case class StreamSingleCodeType(separateRegions: Boolean, eltType: PType) extends SingleCodeType {

  def virtualType: Type = TStream(eltType.virtualType)

  def ti: TypeInfo[_] = classInfo[StreamArgType]

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = {
    val mb = cb.emb
    val xIter = mb.genFieldThisRef[Iterator[java.lang.Long]]("streamInIterator")
    val hasNext = mb.genFieldThisRef[Boolean]("streamInHasNext")
    val next = mb.genFieldThisRef[Long]("streamInNext")

    // this, Region, ...
    val mkIter = coerce[StreamArgType](c)
    val stream = SizedStream.unsized { eltRegion =>
      unfold[Code[Long]](
        (_, k) => Code(
          hasNext := xIter.load().hasNext,
          hasNext.orEmpty(next := xIter.load().next().invoke[Long]("longValue")),
          k(COption(!hasNext, next)))
      ).map(
        rv => EmitCodeBuilder.scopedEmitCode(mb)(cb => EmitCode.present(mb, eltType.loadCheapPCode(cb, (rv)))),
        setup0 = None,
        setup = Some(
          xIter := mkIter.invoke[Region, Region, Iterator[java.lang.Long]](
            "apply", r, eltRegion.code))
      )
    }

    interfaces.SStreamCode(SStream(eltType.sType, separateRegions), stream)
  }
}

case class PTypeReferenceSingleCodeType(pt: PType) extends SingleCodeType {
  def ti: TypeInfo[_] = LongInfo

  def loadToPCode(cb: EmitCodeBuilder, r: Value[Region], c: Code[_]): PCode = pt.loadCheapPCode(cb, coerce[Long](c))

  def virtualType: Type = pt.virtualType
}

object SingleCodePCode {
  def fromPCode(cb: EmitCodeBuilder, pc: PCode, region: Value[Region]): SingleCodePCode = {
    pc.st.virtualType match {
      case TInt32 => SingleCodePCode(Int32SingleCodeType, pc.asInt.intCode(cb))
      case TInt64 => SingleCodePCode(Int64SingleCodeType, pc.asLong.longCode(cb))
      case TFloat32 => SingleCodePCode(Float32SingleCodeType, pc.asFloat.floatCode(cb))
      case TFloat64 => SingleCodePCode(Float64SingleCodeType, pc.asDouble.doubleCode(cb))
      case TBoolean => SingleCodePCode(BooleanSingleCodeType, pc.asBoolean.boolCode(cb))
      case _ =>
        val pt = pc.st.canonicalPType()
        val sct = PTypeReferenceSingleCodeType(pt)
        SingleCodePCode(sct, pt.store(cb, region, pc, deepCopy = false))
    }
  }
}

case class SingleCodePCode(typ: SingleCodeType, code: Code[_])

abstract class PCode extends SCode { self =>

  def st: SType

  def pt: PType

  def code: Code[_]

  def codeTuple(): IndexedSeq[Code[_]]

  def typeInfo: TypeInfo[_] = typeToTypeInfo(pt)

  def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

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

  override def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: PType): PCode =
    castTo(cb, region, destType, false)

  override def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: PType, deepCopy: Boolean): PCode = {
    super.castTo(cb, region, destType, deepCopy).asPCode
  }

  override def copyToRegion(cb: EmitCodeBuilder, region: Value[Region]): PCode = copyToRegion(cb, region, pt)

  override def copyToRegion(cb: EmitCodeBuilder, region: Value[Region], destType: PType): PCode =
    super.copyToRegion(cb, region, destType).asPCode

  def memoize(cb: EmitCodeBuilder, name: String): PValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue

  final def toPCode(cb: EmitCodeBuilder, region: Value[Region]): PCode = this
}

object PCode {
  def apply(pt: PType, code: Code[_]): PCode = pt match {
    case pt: PCanonicalArray  =>
      new SIndexablePointerCode(SIndexablePointer(pt), coerce[Long](code))
    case pt: PCanonicalSet =>
      new SIndexablePointerCode(SIndexablePointer(pt), coerce[Long](code))
    case pt: PCanonicalDict =>
      new SIndexablePointerCode(SIndexablePointer(pt), coerce[Long](code))
    case pt: PSubsetStruct =>
      val ss = pt.sType
      new SSubsetStructCode(ss, PCode(ss.pType.ps, code).asBaseStruct)
    case pt: PCanonicalBaseStruct =>
      new SBaseStructPointerCode(SBaseStructPointer(pt), coerce[Long](code))
    case pt: PCanonicalBinary =>
      new SBinaryPointerCode(SBinaryPointer(pt), coerce[Long](code))
    case pt: PCanonicalShuffle =>
      new SCanonicalShufflePointerCode(SCanonicalShufflePointer(pt),
        new SBinaryPointerCode(SBinaryPointer(pt.representation), coerce[Long](code)))
    case pt: PCanonicalString =>
      new SStringPointerCode(SStringPointer(pt), coerce[Long](code))
    case pt: PCanonicalInterval =>
      new SIntervalPointerCode(SIntervalPointer(pt), coerce[Long](code))
    case pt: PCanonicalLocus =>
      new SCanonicalLocusPointerCode(SCanonicalLocusPointer(pt), coerce[Long](code))
    case pt: PCanonicalCall =>
      new SCanonicalCallCode(pt.required, coerce[Int](code))
    case pt: PCanonicalNDArray =>
      new SNDArrayPointerCode(SNDArrayPointer(pt), coerce[Long](code))
    case pt: PCanonicalStream =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
    case PVoid =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
    case PBoolean(r) =>
      new SBooleanCode(r, coerce[Boolean](code))
    case PInt32(r) =>
      new SInt32Code(r, coerce[Int](code))
    case PInt64(r) =>
      new SInt64Code(r, coerce[Long](code))
    case PFloat32(r) =>
      new SFloat32Code(r, coerce[Float](code))
    case PFloat64(r) =>
      new SFloat64Code(r, coerce[Double](code))
  }

  def _empty: PCode = PVoidCode
}

object PSettable {
  def apply(sb: SettableBuilder, _pt: PType, name: String): PSettable = _pt match {
    case pt: PCanonicalArray =>
      SIndexablePointerSettable(sb, SIndexablePointer(pt), name)
    case pt: PCanonicalSet =>
      SIndexablePointerSettable(sb, SIndexablePointer(pt), name)
    case pt: PCanonicalDict =>
      SIndexablePointerSettable(sb, SIndexablePointer(pt), name)
    case pt: PSubsetStruct =>
      new SSubsetStructSettable(pt.sType, PSettable(sb, pt.ps, name).asInstanceOf[PStructSettable])
    case pt: PCanonicalBaseStruct =>
      SBaseStructPointerSettable(sb, SBaseStructPointer(pt), name)
    case pt: PCanonicalBinary =>
      SBinaryPointerSettable(sb, SBinaryPointer(pt), name)
    case pt: PCanonicalString =>
      SStringPointerSettable(sb, SStringPointer(pt), name)
    case pt: PCanonicalInterval =>
      SIntervalPointerSettable(sb, SIntervalPointer(pt), name)
    case pt: PCanonicalLocus =>
      SCanonicalLocusPointerSettable(sb, SCanonicalLocusPointer(pt), name)
    case pt: PCanonicalCall =>
      SCanonicalCallSettable(sb, name, pt.required)
    case pt: PCanonicalNDArray =>
      SNDArrayPointerSettable(sb, SNDArrayPointer(pt), name)
    case pt: PCanonicalShuffle =>
      SCanonicalShufflePointerSettable(sb, SCanonicalShufflePointer(pt), name)
    case pt: PCanonicalStream =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
    case PVoid =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: PVoid")
    case PBoolean(r) =>
      SBooleanSettable(sb, name, r)
    case PInt32(r) =>
      SInt32Settable(sb, name, r)
    case PInt64(r) =>
      SInt64Settable(sb, name, r)
    case PFloat32(r) =>
      SFloat32Settable(sb, name, r)
    case PFloat64(r) =>
      SFloat64Settable(sb, name, r)
  }
}
