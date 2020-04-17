package is.hail.expr.types.physical

import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.expr.types.virtual.TLocus
import is.hail.variant._

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

  def getLocusObj(): Code[Locus] = Code.invokeStatic2[Locus, String, Int, Locus]("apply",
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
