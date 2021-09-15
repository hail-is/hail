package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.{RInterval, TypeWithRequiredness}
import is.hail.types.physical.PInterval
import is.hail.types.physical.stypes.{EmitType, SCode, SType, SValue}

trait SInterval extends SType {
  def pointType: SType
  def pointEmitType: EmitType
  override def _typeWithRequiredness: TypeWithRequiredness = {
    val pt = pointEmitType.typeWithRequiredness.r
    RInterval(pt, pt)
  }
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

  def codeIncludesStart(): Code[Boolean]

  def codeIncludesEnd(): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): SIntervalValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SIntervalValue
}
