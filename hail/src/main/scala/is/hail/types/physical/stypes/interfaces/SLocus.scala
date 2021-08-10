package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.Code
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.variant.{Locus, ReferenceGenome}

trait SLocus extends SType {
  def rg: ReferenceGenome
  def contigType: SString
}

trait SLocusValue extends SValue {
  def contig(cb: EmitCodeBuilder): SStringCode

  def position(cb: EmitCodeBuilder): Code[Int]

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus] = Code.invokeStatic2[Locus, String, Int, Locus]("apply",
    contig(cb).loadString(), position(cb))

  def structRepr(cb: EmitCodeBuilder): SBaseStructValue
}

trait SLocusCode extends SCode {
  def st: SLocus

  def contig(cb: EmitCodeBuilder): SStringCode

  def position(cb: EmitCodeBuilder): Code[Int]

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String): SLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SLocusValue
}
