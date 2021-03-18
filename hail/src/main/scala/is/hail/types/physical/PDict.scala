package is.hail.types.physical

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.types.virtual.TDict

abstract class PDict extends PContainer {
  lazy val virtualType: TDict = TDict(keyType.virtualType, valueType.virtualType)

  val keyType: PType
  val valueType: PType

  def elementType: PStruct

  override def genNonmissingValue: Gen[Annotation] =
    Gen.buildableOf2[Map](Gen.zip(keyType.genValue, valueType.genValue))
}
