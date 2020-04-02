package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.expr.types.virtual.TLocus
import is.hail.utils._
import is.hail.variant._

object PLocus {
  def apply(rg: ReferenceGenome): PLocus = PCanonicalLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PLocus = PCanonicalLocus(rg.broadcastRG, required)

  def apply(rgBc: BroadcastRG, required: Boolean = false): PLocus = PCanonicalLocus(rgBc, required)
}

abstract class PLocus extends ComplexPType {
  def rgBc: BroadcastRG

  lazy val virtualType: TLocus = TLocus(rgBc)

  def rg: ReferenceGenome

  def contig(value: Long): String

  def contigType: PString

  def position(value: Code[Long]): Code[Int]

  def positionType: PInt32
}

abstract class PLocusValue extends PValue {
  def contig(): PStringCode

  def position(): Value[Int]

  def getLocusObj(): Code[Locus] = Code.invokeStatic[Locus, String, Int, Locus]("apply",
    contig().loadString(), position())
}

abstract class PLocusCode extends PCode {
  def pt: PLocus

  def contig(): PStringCode

  def position(): Code[Int]

  def getLocusObj(): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue
}
