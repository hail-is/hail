package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.variant.Call

case object TCall extends Type {
  def _toPretty = "Call"

  override def pyString(sb: StringBuilder): Unit =
    sb ++= "call"

  val representation: Type = TInt32

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def str(a: Annotation): String =
    if (a == null) "NA" else Call.toString(a.asInstanceOf[Call])

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]], missingEqual)

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
