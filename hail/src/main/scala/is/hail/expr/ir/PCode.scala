package is.hail.expr.ir

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.utils._
import is.hail.expr.types.physical._

trait PValue {
  def pt: PType

  def get: PCode
}

abstract class PIndexableValue extends PValue {
  def loadLength(): Value[Int]

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode
}

abstract class PIndexableSettable extends PIndexableValue with PSettable {
  def loadLength(): Value[Int]

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode
}

object PCanonicalIndexableSettable {
  def apply(sb: SettableBuilder, pt: PContainer, name: String): PCanonicalIndexableSettable = {
    new PCanonicalIndexableSettable(pt,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Int](s"${ name }_length"),
      sb.newSettable[Long](s"${ name }_elems_addr"))
  }
}

class PCanonicalIndexableSettable(
  val pt: PContainer,
  val a: Settable[Long],
  val length: Settable[Int],
  val elementsAddress: Settable[Long]
) extends PIndexableSettable {
  def get: PIndexableCode = new PCanonicalIndexableCode(pt, a)

  def loadLength(): Value[Int] = length

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.memoize(i, "pcindval_i")
    IEmitCode(cb,
      pt.isElementMissing(a, iv),
      pt.elementType.load(elementsAddress + iv.toL * pt.elementByteSize))
  }

  def store(pc: PCode): Code[Unit] = {
    Code(
      a := pc.asInstanceOf[PCanonicalIndexableCode].a,
      length := pt.loadLength(a),
      elementsAddress := pt.firstElementOffset(a, length))
  }
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

    case _ =>
      new PPrimitiveCode(pt, code)
  }

  def _empty: PCode = PCode(PVoid, Code._empty)
}

abstract class PCode { self =>
  def pt: PType

  def code: Code[_]

  def typeInfo: TypeInfo[_] = typeToTypeInfo(pt)

  def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

  def store(mb: EmitMethodBuilder, r: Value[Region], dst: Code[Long]): Code[Unit]

  def allocateAndStore(mb: EmitMethodBuilder, r: Value[Region]): (Code[Unit], Code[Long]) = {
    val dst = mb.newLocal[Long]
    (Code(dst := r.allocate(pt.byteSize, pt.alignment), store(mb, r, dst)), dst)
  }

  def asIndexable: PIndexableCode = asInstanceOf[PIndexableCode]

  def asBaseStruct: PBaseStructCode = asInstanceOf[PBaseStructCode]

  def castTo(mb: EmitMethodBuilder, region: Value[Region], destType: PType): PCode = {
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

trait PSettable extends PValue {
  def store(v: PCode): Code[Unit]

  def load(): PCode = get

  def :=(v: PCode): Code[Unit] = store(v)
}

class PPrimitiveCode(val pt: PType, val code: Code[_]) extends PCode {
  def store(mb: EmitMethodBuilder, r: Value[Region], a: Code[Long]): Code[Unit] =
    Region.storeIRIntermediate(pt)(a, code)

  def memoize(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeImpl(cb, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeFieldImpl(cb, name)
}

abstract class PIndexableCode extends PCode {
  def memoize(cb: EmitCodeBuilder, name: String): PIndexableValue
}

class PCanonicalIndexableCode(val pt: PContainer, val a: Code[Long]) extends PIndexableCode {
  def code: Code[_] = a

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PIndexableValue = {
    val s = PCanonicalIndexableSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PIndexableValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PIndexableValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder, r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}

abstract class PBaseStructCode extends PCode {
  def pt: PBaseStruct

  def isFieldMissing(fieldIdx: Int): Code[Boolean]

  def loadField(fieldIdx: Int): PCode

  def isFieldMissing(fieldName: String): Code[Boolean] = isFieldMissing(pt.fieldIdx(fieldName))

  def isFieldDefined(fieldIdx: Int): Code[Boolean] = !isFieldMissing(fieldIdx)

  def isFieldDefined(fieldName: String): Code[Boolean] = !isFieldMissing(fieldName)

  def loadField(fieldName: String): PCode = loadField(pt.fieldIdx(fieldName))

  def memoize(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeImpl(cb, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeFieldImpl(cb, name)
}

class PCanonicalBaseStructCode(val pt: PCanonicalBaseStruct, val a: Code[Long]) extends PBaseStructCode {
  def code: Code[_] = a

  def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    if (pt.fieldRequired(fieldIdx))
      const(false)
    else
      Region.loadBit(a, pt.missingIdx(fieldIdx).toLong)

  def fieldAddress(fieldIdx: Int): Code[Long] = a + pt.byteOffsets(fieldIdx)

  def loadField(fieldIdx: Int): PCode = pt.fields(fieldIdx).typ.load(fieldAddress(fieldIdx))

  def store(mb: EmitMethodBuilder, r: Value[Region], dst: Code[Long]): Code[Unit] =
    pt.constructAtAddress(mb, dst, r, pt, a, forceDeep = false)
}
