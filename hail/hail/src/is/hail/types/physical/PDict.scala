package is.hail.types.physical

import is.hail.types.physical.stypes.interfaces.SContainer
import is.hail.types.virtual.TDict

abstract class PDict extends PContainer {
  lazy val virtualType: TDict = TDict(keyType.virtualType, valueType.virtualType)

  val keyType: PType
  val valueType: PType

  def sType: SContainer

  def elementType: PStruct
}
