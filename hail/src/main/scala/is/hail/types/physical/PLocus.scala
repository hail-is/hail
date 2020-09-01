package is.hail.types.physical

import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.types.virtual.TLocus
import is.hail.variant._

abstract class PLocus extends PType {
  def rgBc: BroadcastRG

  lazy val virtualType: TLocus = TLocus(rgBc)

  def rg: ReferenceGenome

  def contig(value: Long): String

  def contigType: PString

  def position(value: Code[Long]): Code[Int]

  def position(value: Long): Int

  def positionType: PInt32
}

abstract class PLocusValue extends PValue {
  def contig(cb: EmitCodeBuilder): PStringCode

  def position(cb: EmitCodeBuilder): Code[Int]

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus] = Code.invokeStatic2[Locus, String, Int, Locus]("apply",
    contig(cb).loadString(), position(cb))
}

abstract class PLocusCode extends PCode {
  def pt: PLocus

  def contig(cb: EmitCodeBuilder): PStringCode

  def position(cb: EmitCodeBuilder): Code[Int]

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue
}
