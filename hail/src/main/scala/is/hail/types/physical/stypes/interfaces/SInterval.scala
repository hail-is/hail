package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitSCode}
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SInterval extends SType

trait SIntervalValue extends SValue {
  def includesStart(): Value[Boolean]

  def includesEnd(): Value[Boolean]

  def loadStart(cb: EmitCodeBuilder)(implicit line: LineNumber): IEmitSCode

  def startDefined(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Boolean]

  def loadEnd(cb: EmitCodeBuilder)(implicit line: LineNumber): IEmitSCode

  def endDefined(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Boolean]

  def isEmpty(cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Boolean]
}

trait SIntervalCode extends SCode {
  def includesStart()(implicit line: LineNumber): Code[Boolean]

  def includesEnd()(implicit line: LineNumber): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): SIntervalValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SIntervalValue
}
