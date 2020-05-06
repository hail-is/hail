package is.hail.expr.types.physical

import is.hail.expr.types.virtual.TStream

object PStream {
  def unapply(arg: PType): Option[PType] = arg match {
    case PCanonicalStream(eltType, _) => Some(eltType)
    case _ => None
  }
}

abstract class PStream extends PIterable with PUnrealizable {
  lazy val virtualType: TStream = TStream(elementType.virtualType)

  def _asIdent = s"stream_of_${elementType.asIdent}"
}

abstract class PStreamCode extends PCode with PUnrealizableCode {
  def pt: PStream
}
