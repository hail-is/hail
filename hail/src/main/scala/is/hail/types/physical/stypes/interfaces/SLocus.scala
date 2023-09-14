package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.primitives.{SInt32Value, SInt64Value}
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.variant.{Locus, ReferenceGenome}

trait SLocus extends SType {
  def rg: String
  def contigType: SString
  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()
}

trait SLocusValue extends SValue {
  override def st: SLocus

  def contig(cb: EmitCodeBuilder): SStringValue

  def contigIdx(cb: EmitCodeBuilder): Value[Int]

  def position(cb: EmitCodeBuilder): Value[Int]

  def getLocusObj(cb: EmitCodeBuilder): Value[Locus] =
    cb.memoize(Code.invokeStatic2[Locus, String, Int, Locus]("apply",
      contig(cb).loadString(cb), position(cb)))

  def structRepr(cb: EmitCodeBuilder): SBaseStructValue

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    structRepr(cb).hash(cb)

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = structRepr(cb).sizeToStoreInBytes(cb)
}
