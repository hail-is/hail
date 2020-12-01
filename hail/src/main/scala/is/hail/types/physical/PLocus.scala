package is.hail.types.physical

import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.types.virtual.TLocus
import is.hail.variant._

abstract class PLocus extends ComplexPType {
  def rgBc: BroadcastRG

  lazy val virtualType: TLocus = TLocus(rgBc)

  def rg: ReferenceGenome

  def contig(value: Long): String

  def contigType: PString

  def position(value: Code[Long])(implicit line: LineNumber): Code[Int]

  def positionType: PInt32
}

abstract class PLocusValue extends PValue {
  def contig()(implicit line: LineNumber): PStringCode

  def position(): Value[Int]

  def getLocusObj()(implicit line: LineNumber): Code[Locus] = Code.invokeStatic2[Locus, String, Int, Locus]("apply",
    contig().loadString(), position())
}

abstract class PLocusCode extends PCode {
  def pt: PLocus

  def contig()(implicit line: LineNumber): PStringCode

  def position()(implicit line: LineNumber): Code[Int]

  def getLocusObj()(implicit line: LineNumber): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PLocusValue
}
