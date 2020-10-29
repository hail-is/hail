package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.stypes._
import is.hail.utils._

trait PValue { pValueSelf =>
  def st: SType

  def pt: PType

  def get: PCode

  def value: Value[_] = {
    new Value[Any] {
      override def get: Code[Any] = pValueSelf.get.code
    }
  }
}

trait PSettable extends PValue {

  def store(cb: EmitCodeBuilder, v: PCode): Unit

  def settableTuple(): IndexedSeq[Settable[_]]

  def load(): PCode = get
}

abstract class PCode { self =>

  def st: SType

  def pt: PType

  def code: Code[_]

  def codeTuple(): IndexedSeq[Code[_]]

  def typeInfo: TypeInfo[_] = typeToTypeInfo(pt)

  def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

  def asBoolean: SBooleanCode = asInstanceOf[SBooleanCode]
  def asInt: SInt32Code = asInstanceOf[SInt32Code]
  def asInt32: SInt32Code = asInstanceOf[SInt32Code]
  def asLong: SInt64Code = asInstanceOf[SInt64Code]
  def asInt64: SInt64Code = asInstanceOf[SInt64Code]
  def asFloat: SFloat32Code = asInstanceOf[SFloat32Code]
  def asFloat32: SFloat32Code = asInstanceOf[SFloat32Code]
  def asFloat64: SFloat64Code = asInstanceOf[SFloat64Code]
  def asDouble: SFloat64Code = asInstanceOf[SFloat64Code]
  def asBinary: PBinaryCode = asInstanceOf[PBinaryCode]

  def asIndexable: PIndexableCode = asInstanceOf[PIndexableCode]

  def asBaseStruct: PBaseStructCode = asInstanceOf[PBaseStructCode]

  def asString: PStringCode = asInstanceOf[PStringCode]

  def asInterval: PIntervalCode = asInstanceOf[PIntervalCode]

  def asNDArray: PNDArrayCode = asInstanceOf[PNDArrayCode]

  def asLocus: PLocusCode = asInstanceOf[PLocusCode]

  def asCall: PCallCode = asInstanceOf[PCallCode]

  def asStream: SStreamCode = asInstanceOf[SStreamCode]

  def castTo(cb: EmitCodeBuilder, region: Value[Region], destType: PType, deepCopy: Boolean = false): PCode = {
    destType.sType.coerceOrCopy(cb, region, this, deepCopy)
  }

  def copyToRegion(cb: EmitCodeBuilder, region: Value[Region], destType: PType = pt): PCode =
    destType.sType.coerceOrCopy(cb, region, this, deepCopy = true)

  def memoize(cb: EmitCodeBuilder, name: String): PValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue
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
      new SSubsetStructSettable(pt.sType, PSettable(sb, pt.ps, name).asInstanceOf[SStructSettable])
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
