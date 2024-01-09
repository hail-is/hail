package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.primitives.SInt32Value

trait SCall extends SType {
  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()
}

trait SCallValue extends SValue {
  def unphase(cb: EmitCodeBuilder): SCallValue

  def containsAllele(cb: EmitCodeBuilder, allele: Value[Int]): Value[Boolean]

  def ploidy(cb: EmitCodeBuilder): Value[Int]

  def isPhased(cb: EmitCodeBuilder): Value[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit

  def canonicalCall(cb: EmitCodeBuilder): Value[Int]

  def lgtToGT(cb: EmitCodeBuilder, localAlleles: SIndexableValue, errorID: Value[Int]): SCallValue

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(canonicalCall(cb))
}
