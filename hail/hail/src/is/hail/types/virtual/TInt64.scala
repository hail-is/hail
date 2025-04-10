package is.hail.types.virtual

import is.hail.annotations._


case object TInt64 extends TIntegral {
  def _toPretty = "Int64"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("int64")

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Long]

  override def mkOrdering(missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Long]], missingEqual)

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
