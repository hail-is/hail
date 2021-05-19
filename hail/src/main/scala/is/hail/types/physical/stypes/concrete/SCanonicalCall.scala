package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.{SCall, SCallCode, SCallValue}
import is.hail.types.physical.stypes.{SCode, SSettable, SType}
import is.hail.types.physical.{PCall, PCanonicalCall, PType}
import is.hail.types.virtual.{TCall, Type}
import is.hail.utils._
import is.hail.variant.Genotype


case object SCanonicalCall extends SCall {
  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SCanonicalCall => value
    }
  }

  lazy val virtualType: Type = TCall

  override def castRename(t: Type): SType = this

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(IntInfo)

  def fromSettables(settables: IndexedSeq[Settable[_]]): SCanonicalCallSettable = {
    val IndexedSeq(call: Settable[Int@unchecked]) = settables
    assert(call.ti == IntInfo)
    new SCanonicalCallSettable(call)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SCanonicalCallCode = {
    val IndexedSeq(call: Code[Int@unchecked]) = codes
    assert(call.ti == IntInfo)
    new SCanonicalCallCode(call)
  }

  def canonicalPType(): PType = PCanonicalCall(false)

  def constructFromIntRepr(c: Code[Int]): SCanonicalCallCode = new SCanonicalCallCode(c)
}

object SCanonicalCallSettable {
  def apply(sb: SettableBuilder, name: String): SCanonicalCallSettable =
    new SCanonicalCallSettable(sb.newSettable[Int](s"${ name }_call"))
}

class SCanonicalCallSettable(val call: Settable[Int]) extends SCallValue with SSettable {

  val pt: PCall = PCanonicalCall(false)

  override def canonicalCall(cb: EmitCodeBuilder): Code[Int] = call

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(call, v.asInstanceOf[SCanonicalCallCode].call)

  val st: SCanonicalCall.type = SCanonicalCall

  def get: SCallCode = new SCanonicalCallCode(call)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(call)

  def store(pc: SCode): Code[Unit] = call.store(pc.asInstanceOf[SCanonicalCallCode].call)

  def ploidy(): Code[Int] = get.ploidy()

  def isPhased(): Code[Boolean] = get.isPhased()

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit = {
    val call2 = cb.newLocal[Int]("fea_call2", call >>> 3)
    val p = cb.newLocal[Int]("fea_ploidy", ploidy())
    val j = cb.newLocal[Int]("fea_j")
    val k = cb.newLocal[Int]("fea_k")

    cb.ifx(p.ceq(2), {
      cb.ifx(call2 < Genotype.nCachedAllelePairs, {
        cb.assign(j, Code.invokeScalaObject1[Int, Int](Genotype.getClass, "cachedAlleleJ", call2))
        cb.assign(k, Code.invokeScalaObject1[Int, Int](Genotype.getClass, "cachedAlleleK", call2))
      }, {
        cb.assign(k, (Code.invokeStatic1[Math, Double, Double]("sqrt", const(8d) * call2.toD + 1d) / 2d - 0.5).toI)
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

class SCanonicalCallCode(val call: Code[Int]) extends SCallCode {

  val pt: PCall = PCanonicalCall(false)

  val st: SCanonicalCall.type = SCanonicalCall

  def code: Code[_] = call

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(call)

  def ploidy(): Code[Int] = (call >>> 1) & 0x3

  def isPhased(): Code[Boolean] = (call & 0x1).ceq(1)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SCallValue = {
    val s = SCanonicalCallSettable(sb, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SCallValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SCallValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeInt(dst, call)

  def loadCanonicalRepresentation(cb: EmitCodeBuilder): Code[Int] = call
}
