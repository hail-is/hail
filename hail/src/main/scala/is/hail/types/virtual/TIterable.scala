package is.hail.types.virtual

import is.hail.types.BaseType
import is.hail.utils.FastSeq

abstract class TIterable extends Type {
  def elementType: Type

  override def children = FastSeq(elementType)
}

object TIterable {
  def elementType: BaseType => Type = {
    case t: TIterable => t.elementType
    case t => throw new IllegalArgumentException(
      s"""Could not get element type for $t.
         |  Expected: ${classOf[TIterable].getName}
         |    Actual: ${t.getClass.getName}""".stripMargin
    )
  }
}
