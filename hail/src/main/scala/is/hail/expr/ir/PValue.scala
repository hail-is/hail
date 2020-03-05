package is.hail.expr.ir

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.types.physical.{PCanonicalArray, PCanonicalDict, PCanonicalSet, PContainer, PType, PVoid}

abstract class PSettable[PV <: PValue] {
  def load(): PV

  def store(v: PV): Code[Unit]

  def :=(v: PV): Code[Unit] = store(v)

  def storeAny(v: PValue): Code[Unit] = store(v.asInstanceOf[PV])
}

object PValue {
  def apply(pt: PType, code: Code[_]): PValue = pt match {
    case pt: PCanonicalArray =>
      new PCanonicalIndexableValue(pt, coerce[Long](code))
    case pt: PCanonicalSet =>
      new PCanonicalIndexableValue(pt, coerce[Long](code))
    case pt: PCanonicalDict =>
      new PCanonicalIndexableValue(pt, coerce[Long](code))

    case _ =>
      new PrimitivePValue(pt, code)
  }

  def _empty: PValue = PValue(PVoid, Code._empty)
}

abstract class PValue {
  def pt: PType

  def code: Code[_]

  def typeInfo: TypeInfo[_] = typeToTypeInfo(pt)

  def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }
}

class PrimitivePValue(val pt: PType, val code: Code[_]) extends PValue

abstract class PIndexableValue extends PValue {
  def loadLength(): Code[Int]

  def isElementDefined(i: Code[Int]): Code[Boolean]

  def loadElement(length: Code[Int], i: Code[Int]): PValue

  def loadElement(i: Code[Int]): PValue = loadElement(loadLength(), i)

  def isElementMissing(i: Code[Int]): Code[Boolean] = !isElementDefined(i)
}

class PCanonicalIndexableValue(val pt: PContainer, val a: Code[Long]) extends PIndexableValue {
  def code: Code[_] = a

  def elementType: PType = pt.elementType

  def arrayElementSize: Long = UnsafeUtils.arrayElementSize(elementType)

  def loadLength(): Code[Int] = Region.loadInt(a)

  def nMissingBytes(len: Code[Int]): Code[Int] = (len + 7) >>> 3

  def isElementDefined(i: Code[Int]): Code[Boolean] =
    if (pt.elementType.required)
      const(true)
    else
      !Region.loadBit(a + const(4L), i.toL)

  def elementsOffset(length: Code[Int]): Code[Long] =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(4, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(const(4L) + nMissingBytes(length).toL, elementType.alignment)

  def elementsAddress(length: Code[Int]): Code[Long] = a + elementsOffset(length)

  def elementAddress(length: Code[Int], i: Code[Int]): Code[Long] =
    elementsAddress(length) + i.toL * arrayElementSize

  def loadElement(length: Code[Int], i: Code[Int]): PValue = {
    val elemA = elementAddress(length, i)
    PValue(elementType, Region.loadIRIntermediate(elementType)(elemA))
  }
}
