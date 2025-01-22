package is.hail.types.physical

import is.hail.types.virtual.TSet

abstract class PSet extends PContainer {
  lazy val virtualType: TSet = TSet(elementType.virtualType)
}
