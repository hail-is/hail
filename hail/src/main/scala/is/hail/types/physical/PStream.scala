package is.hail.types.physical

import is.hail.types.virtual.TStream

abstract class PStream extends PIterable with PUnrealizable {
  lazy val virtualType: TStream = TStream(elementType.virtualType)

  def _asIdent = s"stream_of_${elementType.asIdent}"
}

abstract class PStreamCode extends PCode with PUnrealizableCode {
  def pt: PStream
}
