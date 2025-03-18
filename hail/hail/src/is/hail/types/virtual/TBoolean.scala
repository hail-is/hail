package is.hail.types.virtual

import is.hail.annotations._


case object TBoolean extends Type {
  def _toPretty = "Boolean"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("bool")

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Boolean]

  override def _showStr(a: Annotation): String = if (a.asInstanceOf[Boolean]) "True" else "False"

  def parse(s: String): Annotation = s.toBoolean

  override def mkOrdering(missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Boolean]], missingEqual)

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
