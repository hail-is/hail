package is.hail.expr.types.physical

import is.hail.expr.types.virtual.TCall

object PCall {
  def apply(required: Boolean = false): PCall = PCanonicalCall(required)
}

abstract class PCall extends ComplexPType {
  lazy val virtualType: TCall.type = TCall
}
