package is.hail.expr.types.physical

import is.hail.asm4s._
import is.hail.expr.ir.EmitClassBuilder
import is.hail.variant.Genotype

final case class PCanonicalCall(required: Boolean = false) extends PCall {
    def _asIdent = "call"

    override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCCall")

    val representation: PType = PInt32(required)

    def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalCall(required)

    def ploidy(c: Code[Int]): Code[Int] = (c >>> 1) & 0x3

    def isPhased(c: Code[Int]): Code[Boolean] = (c & 0x1).ceq(1)

    def forEachAllele(cb: EmitClassBuilder[_], _c: Code[Int], code: Code[Int] => Code[Unit]): Code[Unit] = {
      val c = cb.genFieldThisRef[Int]()
      val c2 = cb.genFieldThisRef[Int]()
      val p = cb.genFieldThisRef[Int]()
      val j = cb.genFieldThisRef[Int]()
      val k = cb.genFieldThisRef[Int]()

      Code(
        c := _c,
        p := ploidy(c),
        c2 := c >>> 3,
        p.ceq(2).mux(
          Code(
            (c2 < Genotype.nCachedAllelePairs).mux(
              Code(
                j := Code.invokeScalaObject[Int, Int](Genotype.getClass, "cachedAlleleJ", c2),
                k := Code.invokeScalaObject[Int, Int](Genotype.getClass, "cachedAlleleK", c2)
              ),
              Code(
                k := (Code.invokeStatic[Math, Double, Double]("sqrt", const(8d) * c2.toD + 1d) / 2d - 0.5).toI,
                j := c2 - (k * (k + 1) / 2)
              )
            ),
            code(j),
            isPhased(c).mux(
              code(k - j),
              code(k))
          ),
          p.ceq(1).mux(
            code(c2),
            p.cne(0).orEmpty(Code._fatal[Unit](const("invalid ploidy: ").concat(p.toS)))
          )
        )
      )
    }
}
