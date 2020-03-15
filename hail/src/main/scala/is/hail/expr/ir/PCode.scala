package is.hail.expr.ir

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.utils._
import is.hail.expr.types.physical._

abstract class PValue {
  def get: PCode
}

object PCode {
  def apply(pt: PType, code: Code[_]): PCode = pt match {
    case pt: PCanonicalArray =>
      new PCanonicalIndexableCode(pt, coerce[Long](code))
    case pt: PCanonicalSet =>
      new PCanonicalIndexableCode(pt.arrayRep, coerce[Long](code))
    case pt: PCanonicalDict =>
      new PCanonicalIndexableCode(pt.arrayRep, coerce[Long](code))

    case pt: PCanonicalBaseStruct =>
      new PCanonicalBaseStructCode(pt, coerce[Long](code))

    case _ =>
      new PPrimitiveCode(pt, code)
  }

  def _empty: PCode = PCode(PVoid, Code._empty)
}

abstract class PCode {
  def pt: PType

  def code: Code[_]

  def typeInfo: TypeInfo[_] = typeToTypeInfo(pt)

  def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

  def store(mb: EmitMethodBuilder, r: Code[Region], dst: Code[Long]): Code[Unit]

  def allocateAndStore(mb: EmitMethodBuilder, r: Code[Region]): (Code[Unit], Code[Long]) = {
    val dst = mb.newLocal[Long]
    (Code(dst := r.allocate(pt.byteSize, pt.alignment), store(mb, r, dst)), dst)
  }

  def asIndexable: PIndexableCode = asInstanceOf[PIndexableCode]

  def asBaseStruct: PBaseStructCode = asInstanceOf[PBaseStructCode]
}

abstract class PSettable extends PValue {
  def store(v: PCode): Code[Unit]

  def load(): PCode = get

  def :=(v: PCode): Code[Unit] = store(v)
}

class PPrimitiveCode(val pt: PType, val code: Code[_]) extends PCode {
  def store(mb: EmitMethodBuilder, r: Code[Region], a: Code[Long]): Code[Unit] =
    Region.storeIRIntermediate(pt)(a, code)
}

abstract class PIndexableCode extends PCode {
  def loadLength(): Code[Int]

  def isElementDefined(i: Code[Int]): Code[Boolean]

  def loadElement(length: Code[Int], i: Code[Int]): PCode

  def loadElement(i: Code[Int]): PCode

  def isElementMissing(i: Code[Int]): Code[Boolean] = !isElementDefined(i)
}

class PCanonicalIndexableCode(val pt: PCanonicalArray, val a: Code[Long]) extends PIndexableCode {
  def code: Code[_] = a

  def elementType: PType = pt.elementType

  def arrayElementSize: Long = UnsafeUtils.arrayElementSize(elementType)

  def loadLength(): Code[Int] = Region.loadInt(a)

  def nMissingBytes(len: Code[Int]): Code[Int] = (len + 7) >>> 3

  def isElementDefined(i: Code[Int]): Code[Boolean] = pt.isElementDefined(a, i)

  def elementsOffset(length: Code[Int]): Code[Long] = pt.elementsOffset(length)

  def elementAddress(length: Code[Int], i: Code[Int]): Code[Long] = pt.elementOffset(a, length, i)

  def loadElement(length: Code[Int], i: Code[Int]): PCode = {
    elementType.load(Code.memoize(a, "pcindexableval_a") { a =>
      pt.elementOffset(a, length, i)
    })
  }

  def loadElement(i: Code[Int]): PCode = {
    elementType.load(Code.memoize(a, "pcindexableval_a") { a =>
      val length = pt.loadLength(a)
      pt.elementOffset(a, length, i)
    })
  }

  def store(mb: EmitMethodBuilder, r: Code[Region], dst: Code[Long]): Code[Unit] =
    Region.storeAddress(dst, a)
}

abstract class PBaseStructCode extends PCode {
  def pt: PBaseStruct

  def isFieldMissing(fieldIdx: Int): Code[Boolean]

  def loadField(fieldIdx: Int): PCode

  def isFieldMissing(fieldName: String): Code[Boolean] = isFieldMissing(pt.fieldIdx(fieldName))

  def isFieldDefined(fieldIdx: Int): Code[Boolean] = !isFieldMissing(fieldIdx)

  def isFieldDefined(fieldName: String): Code[Boolean] = !isFieldMissing(fieldName)

  def loadField(fieldName: String): PCode = loadField(pt.fieldIdx(fieldName))
}

class PCanonicalBaseStructCode(val pt: PCanonicalBaseStruct, val a: Code[Long]) extends PBaseStructCode {
  def code: Code[_] = a

  def isFieldMissing(fieldIdx: Int): Code[Boolean] = pt.isFieldMissing(a, fieldIdx)

  def fieldAddress(fieldIdx: Int): Code[Long] = pt.fieldOffset(a, fieldIdx)

  def loadField(fieldIdx: Int): PCode = pt.fields(fieldIdx).typ.load(fieldAddress(fieldIdx))

  def store(mb: EmitMethodBuilder, r: Code[Region], dst: Code[Long]): Code[Unit] =
    pt.constructAtAddress(mb, dst, r, pt, a, forceDeep = false)
}
