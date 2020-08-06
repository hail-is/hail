package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.utils._
import is.hail.variant.Genotype

trait PValue { pValueSelf =>
  def pt: PType

  def get: PCode

  def value: Value[_] = {
    new Value[Any] {
      override def get: Code[Any] = pValueSelf.get.code
    }
  }
}

trait PSettable extends PValue {
  def store(v: PCode): Code[Unit]

  def settableTuple(): IndexedSeq[Settable[_]]

  def load(): PCode = get

  def :=(v: PCode): Code[Unit] = store(v)
}

abstract class PCode { self =>
  def pt: PType

  def code: Code[_]

  def codeTuple(): IndexedSeq[Code[_]]

  def typeInfo: TypeInfo[_] = typeToTypeInfo(pt)

  def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit]

  def allocateAndStore(mb: EmitMethodBuilder[_], r: Value[Region]): (Code[Unit], Code[Long]) = {
    val dst = mb.newLocal[Long]()
    (Code(dst := r.allocate(pt.byteSize, pt.alignment), store(mb, r, dst)), dst)
  }

  def asIndexable: PIndexableCode = asInstanceOf[PIndexableCode]

  def asBaseStruct: PBaseStructCode = asInstanceOf[PBaseStructCode]

  def asString: PStringCode = asInstanceOf[PStringCode]

  def asInterval: PIntervalCode = asInstanceOf[PIntervalCode]

  def asNDArray: PNDArrayCode = asInstanceOf[PNDArrayCode]

  def asLocus: PLocusCode = asInstanceOf[PLocusCode]

  def asCall: PCallCode = asInstanceOf[PCallCode]

  def asStream: PCanonicalStreamCode = asInstanceOf[PCanonicalStreamCode]

  def castTo(mb: EmitMethodBuilder[_], region: Value[Region], destType: PType): PCode = {
    PCode(destType,
      destType.copyFromTypeAndStackValue(mb, region, pt, code))
  }

  // this is necessary because Scala doesn't infer the return type of
  // PIndexableCode.memoize if PCode.memoize has a default implementation
  def defaultMemoizeImpl(cb: EmitCodeBuilder, name: String): PValue = {
    new PValue {
      val pt: PType = self.pt

      private val v = cb.newLocalAny(name, code)(typeToTypeInfo(pt))

      def get: PCode = PCode(pt, v)
    }
  }

  def defaultMemoizeFieldImpl(cb: EmitCodeBuilder, name: String): PValue = {
    new PValue {
      val pt: PType = self.pt

      private val v = cb.newFieldAny(name, code)(typeToTypeInfo(pt))

      def get: PCode = PCode(pt, v)
    }
  }

  def memoize(cb: EmitCodeBuilder, name: String): PValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue
}

object PCode {
  def apply(pt: PType, code: Code[_]): PCode = pt match {
    case pt: PCanonicalArray =>
      new PCanonicalIndexableCode(pt, coerce[Long](code))
    case pt: PCanonicalSet =>
      new PCanonicalIndexableCode(pt, coerce[Long](code))
    case pt: PCanonicalDict =>
      new PCanonicalIndexableCode(pt, coerce[Long](code))
    case pt: PSubsetStruct =>
      new PSubsetStructCode(pt, coerce[Long](code))
    case pt: PCanonicalBaseStruct =>
      new PCanonicalBaseStructCode(pt, coerce[Long](code))
    case pt: PCanonicalBinary =>
      new PCanonicalBinaryCode(pt, coerce[Long](code))
    case pt: PCanonicalString =>
      new PCanonicalStringCode(pt, coerce[Long](code))
    case pt: PCanonicalInterval =>
      new PCanonicalIntervalCode(pt, coerce[Long](code))
    case pt: PCanonicalLocus =>
      new PCanonicalLocusCode(pt, coerce[Long](code))
    case pt: PCanonicalCall =>
      new PCanonicalCallCode(pt, coerce[Int](code))
    case pt: PCanonicalNDArray =>
      new PCanonicalNDArrayCode(pt, coerce[Long](code))
    case pt: PCanonicalStream =>
      throw new UnsupportedOperationException(s"Can't PCode.apply unrealizable PType: $pt")
    case _ =>
      new PPrimitiveCode(pt, code)
  }

  def _empty: PCode = PCode(PVoid, Code._empty)
}

object PSettable {
  def apply(sb: SettableBuilder, _pt: PType, name: String): PSettable = _pt match {
    case pt: PCanonicalArray =>
      PCanonicalIndexableSettable(sb, pt, name)
    case pt: PCanonicalSet =>
      PCanonicalIndexableSettable(sb, pt, name)
    case pt: PCanonicalDict =>
      PCanonicalIndexableSettable(sb, pt, name)

    case pt: PCanonicalBaseStruct =>
      PCanonicalBaseStructSettable(sb, pt, name)

    case pt: PCanonicalInterval =>
      PCanonicalIntervalSettable(sb, pt, name)
    case pt: PCanonicalLocus =>
      PCanonicalLocusSettable(sb, pt, name)
    case pt: PCanonicalCall =>
      PCanonicalCallSettable(sb, pt, name)

    case _ => new PSettable {
      val pt: PType = _pt

      private val v = sb.newSettable(name)(typeToTypeInfo(pt))

      def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(v)

      def get: PCode = PCode(pt, v)

      def store(pv: PCode): Code[Unit] = {
        v.storeAny(pv.code)
      }
    }
  }
}
