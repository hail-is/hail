package is.hail.expr.types.physical

import is.hail.asm4s.Code
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.{TStream, Type}

abstract class PStream extends PIterable with PUnrealizable {
  lazy val virtualType: TStream = TStream(elementType.virtualType)

  def _asIdent = s"stream_of_${elementType.asIdent}"
}

abstract class PStreamCode extends PCode with PUnrealizableCode {
  def pt: PStream
}
