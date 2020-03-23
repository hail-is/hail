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
) extends PIndexableValue with PSettable {
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

    case pt: PBinary =>
      new PCanonicalBinaryCode(pt, coerce[Long](code))

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

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit]

  def allocateAndStore(mb: EmitMethodBuilder[_], r: Value[Region]): (Code[Unit], Code[Long]) = {
    val dst = mb.newLocal[Long]()
    (Code(dst := r.allocate(pt.byteSize, pt.alignment), store(mb, r, dst)), dst)
  }

  def asIndexable: PIndexableCode = asInstanceOf[PIndexableCode]

  def asBaseStruct: PBaseStructCode = asInstanceOf[PBaseStructCode]

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

trait PSettable extends PValue {
  def store(v: PCode): Code[Unit]

  def load(): PCode = get

  def :=(v: PCode): Code[Unit] = store(v)
}

class PPrimitiveCode(val pt: PType, val code: Code[_]) extends PCode {
  def store(mb: EmitMethodBuilder[_], r: Value[Region], a: Code[Long]): Code[Unit] =
    Region.storeIRIntermediate(pt)(a, code)

  def memoize(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeImpl(cb, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeFieldImpl(cb, name)
}

abstract class PIndexableCode extends PCode {
  def loadLength(): Code[Int]

  def memoize(cb: EmitCodeBuilder, name: String): PIndexableValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PIndexableValue
}

class PCanonicalIndexableCode(val pt: PContainer, val a: Code[Long]) extends PIndexableCode {
  def code: Code[_] = a

  def loadLength(): Code[Int] = pt.loadLength(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PIndexableValue = {
    val s = PCanonicalIndexableSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PIndexableValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PIndexableValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}

abstract class PBaseStructValue extends PValue {
  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode

  def loadField(cb: EmitCodeBuilder, fieldName: String): IEmitCode = loadField(cb, pt.asInstanceOf[PBaseStruct].fieldIdx(fieldName))
}

object PCanonicalBaseStructSettable {
  def apply(cb: EmitCodeBuilder, pt: PBaseStruct, name: String, sb: SettableBuilder): PCanonicalBaseStructSettable = {
    new PCanonicalBaseStructSettable(pt, sb.newSettable(name))
  }
}

class PCanonicalBaseStructSettable(
  val pt: PBaseStruct,
  val a: Settable[Long]
) extends PBaseStructValue with PSettable {
  def get: PBaseStructCode = new PCanonicalBaseStructCode(pt, a)

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    IEmitCode(cb,
      pt.isFieldMissing(a, fieldIdx),
      pt.fields(fieldIdx).typ.load(pt.fieldOffset(a, fieldIdx)))
  }

  def store(pv: PCode): Code[Unit] = {
    a := pv.asInstanceOf[PCanonicalBaseStructCode].a
  }
}

abstract class PBaseStructCode extends PCode {
  def pt: PBaseStruct

  def memoize(cb: EmitCodeBuilder, name: String): PBaseStructValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PBaseStructValue
}

class PCanonicalBaseStructCode(val pt: PBaseStruct, val a: Code[Long]) extends PBaseStructCode {
  def code: Code[_] = a

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PBaseStructValue = {
    val s = PCanonicalBaseStructSettable(cb, pt, name, sb)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    pt.constructAtAddress(mb, dst, r, pt, a, deepCopy = false)
}

abstract class PBinaryCode extends PCode {
  def loadLength(): Code[Int]

  def bytesAddress(): Code[Long]

  def loadBytes(): Code[Array[Byte]]
}

class PCanonicalBinaryCode(val pt: PBinary, a: Code[Long]) extends PBinaryCode {
  def code: Code[_] = a

  def loadLength(): Code[Int] = pt.loadLength(a)

  def bytesAddress(): Code[Long] = pt.bytesAddress(a)

  def loadBytes(): Code[Array[Byte]] = pt.loadBytes(a)

  def memoize(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeImpl(cb, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeFieldImpl(cb, name)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}
