package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.utils._
import is.hail.variant.Genotype

trait PValue {
  def pt: PType

  def get: PCode
}

trait PSettable extends PValue {
  def store(v: PCode): Code[Unit]

  def load(): PCode = get

  def :=(v: PCode): Code[Unit] = store(v)
}

abstract class PCode { self =>
  def pt: PType

  def code: Code[_]

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

  def asLocus: PLocusCode = asInstanceOf[PLocusCode]

  def asCall: PCallCode = asInstanceOf[PCallCode]

  def castTo(mb: EmitMethodBuilder[_], region: Value[Region], destType: PType): PCode = {
    PCode(destType,
      destType.copyFromTypeAndStackValue(mb, region, pt, code))
  }

  // this is necessary because Scala doesn't infer the return type of
  // PIndexableCode.memoize if PCode.memoize has a default implementation
  def defaultMemoizeImpl(cb: EmitCodeBuilder, name: String): PValue = {
    new PValue {
      val pt: PType = self.pt

      private val v = cb.memoizeAny(code, name)(typeToTypeInfo(pt))

      def get: PCode = PCode(pt, v)
    }
  }

  def defaultMemoizeFieldImpl(cb: EmitCodeBuilder, name: String): PValue = {
    new PValue {
      val pt: PType = self.pt

      private val v = cb.memoizeFieldAny(code, name)(typeToTypeInfo(pt))

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

    case pt: PCanonicalBaseStruct =>
      new PCanonicalBaseStructCode(pt, coerce[Long](code))

    case pt: PBinary =>
      new PCanonicalBinaryCode(pt, coerce[Long](code))
    case pt: PString =>
      new PCanonicalStringCode(pt, coerce[Long](code))
    case pt: PInterval =>
      new PCanonicalIntervalCode(pt, coerce[Long](code))
    case pt: PCanonicalLocus =>
      new PCanonicalLocusCode(pt, coerce[Long](code))
    case pt: PCall =>
      new PCanonicalCallCode(pt, coerce[Int](code))

    case _ =>
      new PPrimitiveCode(pt, code)
  }

  def _empty: PCode = PCode(PVoid, Code._empty)
}
