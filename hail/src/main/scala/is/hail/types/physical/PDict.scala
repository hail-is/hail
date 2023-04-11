package is.hail.types.physical

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.types.physical.stypes.interfaces.SContainer
import is.hail.types.virtual.TDict

abstract class PDict extends PContainer {
  lazy val virtualType: TDict = TDict(keyType.virtualType, valueType.virtualType)

  val keyType: PType
  val valueType: PType

  def sType: SContainer

  def elementType: PStruct

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
    Gen.buildableOf2[Map](Gen.zip(keyType.genValue(sm), valueType.genValue(sm)))
}
