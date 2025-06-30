package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager

case object TString extends Type {
  def _toPretty = "String"

  override def pyString(sb: StringBuilder): Unit =
    sb ++= "str"

  override def _showStr(a: Annotation): String = "\"" + a.asInstanceOf[String] + "\""

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[String]

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[String]], missingEqual)

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
