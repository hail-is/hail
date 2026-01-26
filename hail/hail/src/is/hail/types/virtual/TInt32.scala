package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager

case object TInt32 extends TIntegral {
  override def _toPretty = "Int32"

  override def pyString(sb: StringBuilder): Unit =
    sb ++= "int32"

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]], missingEqual)

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
