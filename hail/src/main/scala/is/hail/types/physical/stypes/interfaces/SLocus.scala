package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.variant.Locus

trait SLocus extends SType

trait SLocusValue extends SValue {
  def contig(cb: EmitCodeBuilder)(implicit line: LineNumber): SStringCode

  def position(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Int]

  def getLocusObj(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Locus] = Code.invokeStatic2[Locus, String, Int, Locus]("apply",
    contig(cb).loadString(), position(cb))
}

trait SLocusCode extends SCode {
  def contig(cb: EmitCodeBuilder)(implicit line: LineNumber): SStringCode

  def position(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Int]

  def getLocusObj(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): SLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): SLocusValue
}
