package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.types.physical.PVoid

case object TVoid extends Type {
  override def _toPretty = "Void"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("void")

  def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering = null

  override def scalaClassTag: scala.reflect.ClassTag[_ <: AnyRef] =
    throw new UnsupportedOperationException("No ClassTag for Void")

  override def _typeCheck(a: Any): Boolean =
    throw new UnsupportedOperationException("No elements of Void")

  override def isRealizable = false
}
