package is.hail.expr.types

import is.hail.expr.types.physical.PContainer
import is.hail.utils._

abstract class TContainer extends Type {
  def physicalType: PContainer

  def elementType: Type

  override def children = FastSeq(elementType)
}
