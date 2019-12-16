package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, coerce}
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TLocus
import is.hail.utils._
import is.hail.variant._

object PLocus {
  def apply(rg: ReferenceGenome): PLocus = PCanonicalLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PLocus = PCanonicalLocus(rg.broadcastRG, required)

  def apply(rgBc: BroadcastRG, required: Boolean = false): PLocus = PCanonicalLocus(rgBc, required)
}

abstract class PLocus extends PType {
  def rgBc: BroadcastRG

  lazy val virtualType: TLocus = TLocus(rgBc, required)

  def representation: PStruct

  def rg: ReferenceGenome

  def contig(region: Code[Region], off: Code[Long]): Code[Long]

  def position(region: Code[Region], off: Code[Long]): Code[Int]

  def copy(required: Boolean): PLocus
}
