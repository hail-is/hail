package is.hail.types.physical

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.types.virtual.TSet

abstract class PSet extends PContainer {
  lazy val virtualType: TSet = TSet(elementType.virtualType)

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
    Gen.buildableOf[Set](elementType.genValue(sm))
}
