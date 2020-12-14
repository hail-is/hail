package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SCall extends SType

trait SCallValue extends SValue {
  def ploidy()(implicit line: LineNumber): Code[Int]

  def isPhased()(implicit line: LineNumber): Code[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit
}

trait SCallCode extends SCode {
  def ploidy()(implicit line: LineNumber): Code[Int]

  def isPhased()(implicit line: LineNumber): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): SCallValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SCallValue
}
