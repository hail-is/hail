package is.hail.types.virtual

import is.hail.annotations.ExtendedOrdering

case object TVoid extends Type {
  override def _toPretty = "Void"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("void")

  override def mkOrdering(missingEqual: Boolean): ExtendedOrdering = null

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[Unit]

  override def isRealizable = false

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
