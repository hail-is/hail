package is.hail.expr.types.physical
import is.hail.variant.ReferenceGenome

import is.hail.annotations._
import is.hail.asm4s.{Code, coerce}
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TLocus
import is.hail.utils._
import is.hail.variant._


object PCanonicalLocus {
  def apply(rg: ReferenceGenome): PLocus = PCanonicalLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PLocus = PCanonicalLocus(rg.broadcastRG, required)

  def representation(required: Boolean = false): PStruct = PStruct(
    required,
    "contig" -> PString(required = true),
    "position" -> PInt32(required = true))

  def schemaFromRG(rg: Option[ReferenceGenome], required: Boolean = false): PType = rg match {
    case Some(ref) => PCanonicalLocus(ref)
    case None => representation(required)
  }
}

final case class PCanonicalLocus(rgBc: BroadcastRG, required: Boolean = false) extends PLocus with ComplexPType {
    def rg: ReferenceGenome = rgBc.value

    def _asIdent = "locus"
    def _toPretty = s"Locus($rg)"

    override def pyString(sb: StringBuilder): Unit = {
      sb.append("locus<")
      sb.append(prettyIdentifier(rg.name))
      sb.append('>')
    }

    def copy(required: Boolean = this.required) = PCanonicalLocus(this.rgBc, required)

    val representation: PStruct = PCanonicalLocus.representation(required)

    def contig(region: Code[Region], off: Code[Long]): Code[Long] = representation.loadField(region, off, 0)

    def position(region: Code[Region], off: Code[Long]): Code[Int] = Region.loadInt(representation.loadField(region, off, 1))

}
