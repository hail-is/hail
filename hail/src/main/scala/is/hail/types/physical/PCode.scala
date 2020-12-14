package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.PVoidCode
import is.hail.types.physical.stypes.primitives._

trait PValue extends SValue { pValueSelf =>
  def pt: PType

  def get(implicit line: LineNumber): PCode

  def value: Value[_] = {
    new Value[Any] {
      override def get(implicit line: LineNumber): Code[Any] = pValueSelf.get.code
    }
  }
}

trait PSettable extends PValue {

  def store(cb: EmitCodeBuilder, v: PCode): Unit

  def settableTuple(): IndexedSeq[Settable[_]]

  def load()(implicit line: LineNumber): PCode = get
}

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

  override def copyToRegion(cb: EmitCodeBuilder, region: Value[Region]): PCode =
    copyToRegion(cb, region, pt)

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
