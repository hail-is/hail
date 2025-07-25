package is.hail.types.virtual

import is.hail.annotations.ExtendedOrdering
import is.hail.backend.HailStateManager
import is.hail.macros.void

case object TVoid extends Type {
  override def _toPretty = "Void"

  override def pyString(sb: StringBuilder): Unit =
    void(sb ++= "void")

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering = null

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[Unit]

  override def isRealizable = false

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
