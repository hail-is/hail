package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TCall
import is.hail.variant.Genotype

case object PCallOptional extends PCall(false)

case object PCallRequired extends PCall(true)

class PCall(override val required: Boolean) extends ComplexPType {
  lazy val virtualType: TCall = TCall(required)

  def _toPretty = "Call"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("call")
  }

  val representation: PType = PCall.representation(required)

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    PInt32().codeOrdering(mb)
  }

  def ploidy(c: Code[Int]): Code[Int] = (c >>> 1) & 0x3

  def isPhased(c: Code[Int]): Code[Boolean] = (c & 0x1).ceq(1)

  def forEachAllele(mb: EmitMethodBuilder, _c: Code[Int], code: Code[Int] => Code[Unit]): Code[Unit] = {
    val c = mb.newLocal[Int]
    val p = mb.newLocal[Int]
    val j = mb.newLocal[Int]
    val k = mb.newLocal[Int]

    Code(
      c := _c,
      p := ploidy(c),
      p.ceq(2).mux(
        Code(
          (c < Genotype.nCachedAllelePairs).mux(
            Code(
              j := Code.invokeScalaObject[Int, Int](Genotype.getClass, "cachedAlleleJ", c >>> 3),
              k := Code.invokeScalaObject[Int, Int](Genotype.getClass, "cachedAlleleK", c >>> 3)
            ),
            Code(
              k := (Code.invokeStatic[Math, Double, Double]("sqrt", const(8d) * (c >>> 3).toD + const(1)) / 2d - 0.5).toI,
              j := (c >>> 3) - (k * (k + 1) / 2)
            )
          ),
          code(j),
          isPhased(c).mux(
            code(k - j),
            code(k))
        ),
        p.ceq(1).mux(
          code(c >>> 3),
          p.cne(0).orEmpty(Code._fatal(const("invalid ploidy: ").concat(p.toS)))
        )
      )
    )
  }
}

object PCall {
  def apply(required: Boolean = false): PCall = if (required) PCallRequired else PCallOptional

  def unapply(t: PCall): Option[Boolean] = Option(t.required)

  def representation(required: Boolean = false): PType = PInt32(required)
}
