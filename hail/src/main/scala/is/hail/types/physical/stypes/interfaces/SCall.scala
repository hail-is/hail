package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.{PCode, PValue}

trait SCall extends SType

trait SCallValue extends SValue {
  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit

  def canonicalCall(cb: EmitCodeBuilder): Code[Int]
}

trait SCallCode extends SCode {
  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): SCallValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SCallValue

  def loadCanonicalRepresentation(cb: EmitCodeBuilder): Code[Int]
}
