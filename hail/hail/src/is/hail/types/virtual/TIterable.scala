package is.hail.types.virtual

import is.hail.collection.FastSeq
import is.hail.types.BaseType

abstract class TIterable extends Type {
  def elementType: Type

  override def children = FastSeq(elementType)
}

object TIterable {
  def elementType(t: BaseType): Type =
    t.asInstanceOf[TIterable].elementType
}
