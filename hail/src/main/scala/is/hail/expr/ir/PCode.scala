package is.hail.expr.ir

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.variant.Genotype

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
    case pt: PString =>
      new PCanonicalStringCode(pt, coerce[Long](code))
    case pt: PInterval =>
      new PCanonicalIntervalCode(pt, coerce[Long](code))
    case pt: PLocus =>
      new PCanonicalLocusCode(pt, coerce[Long](code))
    case pt: PCall =>
      new PCanonicalCallCode(pt, coerce[Int](code))

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

  def asString: PStringCode = asInstanceOf[PStringCode]

  def asInterval: PIntervalCode = asInstanceOf[PIntervalCode]

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
  def pt: PContainer

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
  def pt: PBinary

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

abstract class PStringCode extends PCode {
  def pt: PString

  def loadLength(): Code[Int]

  def bytesAddress(): Code[Long]

  def loadString(): Code[String]
}

class PCanonicalStringCode(val pt: PString, a: Code[Long]) extends PStringCode {
  def code: Code[_] = a

  def loadLength(): Code[Int] = pt.loadLength(a)

  def bytesAddress(): Code[Long] = pt.bytesAddress(a)

  def loadString(): Code[String] = pt.loadString(a)

  def memoize(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeImpl(cb, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeFieldImpl(cb, name)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}

abstract class PLocusValue extends PValue {
  def contig(): PStringCode

  def position(): Value[Int]
}

object PCanonicalLocusSettable {
  def apply(sb: SettableBuilder, pt: PLocus, name: String): PCanonicalLocusSettable = {
    new PCanonicalLocusSettable(pt,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Long](s"${ name }_contig"),
      sb.newSettable[Int](s"${ name }_position"))

  }
}

class PCanonicalLocusSettable(
  val pt: PLocus,
  val a: Settable[Long],
  _contig: Settable[Long],
  val position: Settable[Int]
) extends PLocusValue with PSettable {
  def get = new PCanonicalLocusCode(pt, a)

  def store(pc: PCode): Code[Unit] = {
    Code(
      a := pc.asInstanceOf[PCanonicalLocusCode].a,
      _contig := pt.contig(a),
      position := pt.position(a))
  }

  def contig(): PStringCode = new PCanonicalStringCode(pt.contigType, _contig)
}

abstract class PLocusCode extends PCode {
  def pt: PLocus

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue
}

class PCanonicalLocusCode(val pt: PLocus, val a: Code[Long]) extends PLocusCode {
  def code: Code[_] = a

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PLocusValue = {
    val s = PCanonicalLocusSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}

object PCallValue {
  def apply(pt: PCall, call: Settable[_]): PCallValue = new PCanonicalCallSettable(pt, coerce[Int](call))
}

abstract class PCallValue extends PValue {
  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit
}

object PCanonicalCallSettable {
  def apply(sb: SettableBuilder, pt: PCall, name: String): PCanonicalCallSettable =
    new PCanonicalCallSettable(pt, sb.newSettable[Int](s"${ name }_call"))
}

class PCanonicalCallSettable(val pt: PCall, call: Settable[Int]) extends PCallValue with PSettable {
  def get: PCallCode = new PCanonicalCallCode(pt, call)

  def store(pc: PCode): Code[Unit] = call.store(pc.asInstanceOf[PCanonicalCallCode].call)

  def ploidy(): Code[Int] = get.ploidy()

  def isPhased(): Code[Boolean] = get.isPhased()

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit = {
    val call2 = cb.memoize[Int](call >>> 3, "fea_call2")
    val p = cb.memoize[Int](ploidy(), "fea_ploidy")
    val j = cb.localBuilder.newSettable[Int]("fea_j")
    val k = cb.localBuilder.newSettable[Int]("fea_k")

    cb.ifx(p.ceq(2), {
      cb.ifx(call2 < Genotype.nCachedAllelePairs, {
        cb.assign(j, Code.invokeScalaObject[Int, Int](Genotype.getClass, "cachedAlleleJ", call2))
        cb.assign(k, Code.invokeScalaObject[Int, Int](Genotype.getClass, "cachedAlleleK", call2))
      }, {
        cb.assign(k, (Code.invokeStatic[Math, Double, Double]("sqrt", const(8d) * call2.toD + 1d) / 2d - 0.5).toI)
        cb.assign(j, call2 - (k * (k + 1) / 2))
      })
      alleleCode(j)
      cb.ifx(isPhased(), cb.assign(k, k - j))
      alleleCode(k)
      }, {
        cb.ifx(p.ceq(1),
          alleleCode(call2),
          cb.ifx(p.cne(0),
            cb.append(Code._fatal[Unit](const("invalid ploidy: ").concat(p.toS)))))
      })
  }
}

abstract class PCallCode extends PCode {
  def pt: PCall

  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): PCallValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PCallValue
}

class PCanonicalCallCode(val pt: PCall, val call: Code[Int]) extends PCallCode {
  def code: Code[_] = call

  def ploidy(): Code[Int] = (call >>> 1) & 0x3

  def isPhased(): Code[Boolean] = (call & 0x1).ceq(1)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PCallValue = {
    val s = PCanonicalCallSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PCallValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PCallValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeInt(dst, call)
}

abstract class PIntervalValue extends PValue {
  def includesStart(): Value[Boolean]

  def includesEnd(): Value[Boolean]

  def loadStart(cb: EmitCodeBuilder): IEmitCode

  def loadEnd(cb: EmitCodeBuilder): IEmitCode
}

object PCanonicalIntervalSettable {
  def apply(sb: SettableBuilder, pt: PInterval, name: String): PCanonicalIntervalSettable = {
    new PCanonicalIntervalSettable(pt,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Boolean](s"${ name }_includes_start"),
      sb.newSettable[Boolean](s"${ name }_includes_end"))
  }
}

class PCanonicalIntervalSettable(
  val pt: PInterval,
  a: Settable[Long],
  val includesStart: Settable[Boolean],
  val includesEnd: Settable[Boolean]
) extends PIntervalValue with PSettable {
  def get: PIntervalCode = new PCanonicalIntervalCode(pt, a)

  def loadStart(cb: EmitCodeBuilder): IEmitCode =
    IEmitCode(cb,
      !(pt.startDefined(a)),
      pt.pointType.load(pt.loadStart(a)))

  def loadEnd(cb: EmitCodeBuilder): IEmitCode =
    IEmitCode(cb,
      !(pt.endDefined(a)),
      pt.pointType.load(pt.loadEnd(a)))

  def store(pc: PCode): Code[Unit] = {
    Code(
      a := pc.asInstanceOf[PCanonicalIntervalCode].a,
      includesStart := pt.includesStart(a.load()),
      includesEnd := pt.includesEnd(a.load()))
  }
}

abstract class PIntervalCode extends PCode {
  def pt: PInterval

  def memoize(cb: EmitCodeBuilder, name: String): PIntervalValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PIntervalValue
}

class PCanonicalIntervalCode(val pt: PInterval, val a: Code[Long]) extends PIntervalCode {
  def code: Code[_] = a

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PIntervalValue = {
    val s = PCanonicalIntervalSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PIntervalValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PIntervalValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    pt.constructAtAddress(mb, dst, r, pt, a, deepCopy = false)
}
