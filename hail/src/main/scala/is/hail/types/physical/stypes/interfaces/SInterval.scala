package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitSCode}
import is.hail.types.physical.PInterval
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SInterval extends SType {
  def pointType: SType
}

trait SIntervalValue extends SValue {
  def st: SInterval

  def includesStart(): Value[Boolean]

  def includesEnd(): Value[Boolean]

  def loadStart(cb: EmitCodeBuilder): IEmitSCode

  def startDefined(cb: EmitCodeBuilder): Code[Boolean]

  def loadEnd(cb: EmitCodeBuilder): IEmitSCode

  def endDefined(cb: EmitCodeBuilder): Code[Boolean]

  def isEmpty(cb: EmitCodeBuilder): Code[Boolean]
}

trait SIntervalCode extends SCode {
  def st: SInterval

  def includesStart(): Code[Boolean]

  def includesEnd(): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): SIntervalValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SIntervalValue
}
