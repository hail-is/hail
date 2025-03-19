package is.hail.types.virtual

import is.hail.annotations._


case object TBinary extends Type {
  def _toPretty = "Binary"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  def mkOrdering(_missingEqual: Boolean = true): ExtendedOrdering =
    ExtendedOrdering.iterableOrdering(new ExtendedOrdering {
      val missingEqual = _missingEqual

      override def compareNonnull(x: Any, y: Any): Int =
        java.lang.Integer.compare(
          java.lang.Byte.toUnsignedInt(x.asInstanceOf[Byte]),
          java.lang.Byte.toUnsignedInt(y.asInstanceOf[Byte]),
        )
    })

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
