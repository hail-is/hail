package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.primitives.SInt32Code
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.{RPrimitive, TypeWithRequiredness}

trait SCall extends SType {
  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()
}

trait SCallValue extends SValue {
  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit

  def canonicalCall(cb: EmitCodeBuilder): Value[Int]

  def lgtToGT(cb: EmitCodeBuilder, localAlleles: SIndexableValue, errorID: Value[Int]): SCallValue

  override def hash(cb: EmitCodeBuilder): SInt32Code = {
    new SInt32Code(canonicalCall(cb))
  }
}

trait SCallCode extends SCode {
  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): SCallValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SCallValue

  def loadCanonicalRepresentation(cb: EmitCodeBuilder): Code[Int]
}
