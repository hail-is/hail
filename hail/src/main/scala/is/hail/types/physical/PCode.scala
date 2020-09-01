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

  def asBoolean: PBooleanCode = asInstanceOf[PBooleanCode]
  def asInt: PInt32Code = asInstanceOf[PInt32Code]
  def asInt32: PInt32Code = asInstanceOf[PInt32Code]
  def asLong: PInt64Code = asInstanceOf[PInt64Code]
  def asInt64: PInt64Code = asInstanceOf[PInt64Code]
  def asFloat: PFloat32Code = asInstanceOf[PFloat32Code]
  def asFloat32: PFloat32Code = asInstanceOf[PFloat32Code]
  def asFloat64: PFloat64Code = asInstanceOf[PFloat64Code]
  def asDouble: PFloat64Code = asInstanceOf[PFloat64Code]
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
      new SCanonicalCallCode(coerce[Int](code))
    case pt: PCanonicalNDArray =>
      new SNDArrayPointerCode(SNDArrayPointer(pt), coerce[Long](code))
    case pt: PCanonicalStream =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
    case PVoid =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
    case PBoolean(r) =>
      new SCanonicalBooleanCode(coerce[Boolean](code))
    case PInt32(r) =>
      new SCanonicalInt32Code(coerce[Int](code))
    case PInt64(r) =>
      new SCanonicalInt64Code(coerce[Long](code))
    case PFloat32(r) =>
      new SCanonicalFloat32Code(coerce[Float](code))
    case PFloat64(r) =>
      new SCanonicalFloat64Code(coerce[Double](code))
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
    //    case pt: PSubsetStruct =>
    //      new PSubsetStructCode(pt, coerce[Long](code))
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
      SCanonicalCallSettable(sb, name)
    case pt: PCanonicalNDArray =>
      SNDArrayPointerSettable(sb, SNDArrayPointer(pt), name)
    case pt: PCanonicalStream =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
    case PVoid =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: PVoid")
    case PBoolean(r) =>
      SCanonicalBooleanSettable(sb, name)
    case PInt32(r) =>
      SCanonicalInt32Settable(sb, name)
    case PInt64(r) =>
      SCanonicalInt64Settable(sb, name)
    case PFloat32(r) =>
      SCanonicalFloat32Settable(sb, name)
    case PFloat64(r) =>
      SCanonicalFloat64Settable(sb, name)
  }
}
