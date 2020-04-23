package is.hail.expr.types.physical

import is.hail.expr.types.virtual.TStream

abstract class PStream extends PIterable with PUnrealizable {
  lazy val virtualType: TStream = TStream(elementType.virtualType)

  def _asIdent = s"stream_of_${elementType.asIdent}"
}

abstract class PStreamCode extends PCode with PUnrealizableCode {
  def pt: PStream
}
