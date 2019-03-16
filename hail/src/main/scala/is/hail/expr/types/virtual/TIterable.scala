package is.hail.expr.types.virtual

import is.hail.expr.types.physical.PIterable
import is.hail.utils.FastSeq

abstract class TIterable extends Type {
  def physicalType: PIterable

  def elementType: Type

  override def children = FastSeq(elementType)
}
