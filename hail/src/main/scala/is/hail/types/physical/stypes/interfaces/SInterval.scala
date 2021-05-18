package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.PInterval
import is.hail.types.physical.stypes.{EmitType, SCode, SType, SValue}

trait SInterval extends SType {
  def pointType: SType
  def pointEmitType: EmitType
}

trait SIntervalValue extends SValue {
  def st: SInterval

  def includesStart(): Value[Boolean]

  def includesEnd(): Value[Boolean]

  def loadStart(cb: EmitCodeBuilder): IEmitCode

  def startDefined(cb: EmitCodeBuilder): Code[Boolean]

  def loadEnd(cb: EmitCodeBuilder): IEmitCode

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
