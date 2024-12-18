package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager
import is.hail.check.Gen

case object TVoid extends Type {
  override def _toPretty = "Void"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("void")

  def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering = null

  override def scalaClassTag: scala.reflect.ClassTag[_ <: AnyRef] =
    throw new UnsupportedOperationException("No ClassTag for Void")

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[Unit]

  override def isRealizable = false

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
