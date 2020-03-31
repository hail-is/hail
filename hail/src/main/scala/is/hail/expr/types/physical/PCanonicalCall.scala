package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.variant.Genotype

final case class PCanonicalCall(required: Boolean = false) extends PCall {
    def _asIdent = "call"

    override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCCall")

    val representation: PType = PInt32(required)

    def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
      assert(other isOfType this)
      PInt32().codeOrdering(mb)
    }

    def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalCall(required)
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
