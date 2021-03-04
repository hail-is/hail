package is.hail.types.physical

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.types.virtual.TSet

abstract class PSet extends PContainer {
  lazy val virtualType: TSet = TSet(elementType.virtualType)

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf[Set](elementType.genValue)
}
