package is.hail.types.virtual

import is.hail.annotations._

case object TString extends Type {
  override def _toPretty = "String"

  override def pyString(sb: StringBuilder): Unit =
    sb ++= "str"

  override def _showStr(a: Annotation): String = "\"" + a.asInstanceOf[String] + "\""

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[String]

  override def mkOrdering(missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[String]], missingEqual)

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
