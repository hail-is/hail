package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.primitives.SInt32Code
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.variant.{Locus, ReferenceGenome}

trait SLocus extends SType {
  def rg: ReferenceGenome
  def contigType: SString
  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()
}

trait SLocusValue extends SValue {
  override def st: SLocus

  def contig(cb: EmitCodeBuilder): SStringValue

  def contigLong(cb: EmitCodeBuilder): Value[Long]

  def position(cb: EmitCodeBuilder): Value[Int]

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus] = Code.invokeStatic2[Locus, String, Int, Locus]("apply",
    contig(cb).loadString(cb), position(cb))

  def structRepr(cb: EmitCodeBuilder): SBaseStructValue

  override def hash(cb: EmitCodeBuilder): SInt32Code = structRepr(cb).hash(cb)
}

trait SLocusCode extends SCode {
  def st: SLocus

  def contig(cb: EmitCodeBuilder): SStringCode

  def position(cb: EmitCodeBuilder): Code[Int]

  def getLocusObj(cb: EmitCodeBuilder): Code[Locus]

  def memoize(cb: EmitCodeBuilder, name: String): SLocusValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SLocusValue
}
