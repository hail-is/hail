package is.hail.types.physical

import is.hail.types.virtual.TCall

abstract class PCall extends PType {
  lazy val virtualType: TCall.type = TCall
}
