package is.hail.expr.types

import is.hail.expr.types.physical.PContainer
import is.hail.utils._

abstract class TContainer extends Type {
  def physicalType: PContainer

  def elementType: Type

  def elementByteSize: Long

  override def byteSize: Long = 8

  def contentsAlignment: Long

  override def children = FastSeq(elementType)
}
