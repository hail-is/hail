package is.hail.types.virtual

import is.hail.annotations.Annotation
import is.hail.backend.HailStateManager
import is.hail.check.Gen

case object TRNGState extends Type {
  override def _toPretty = "RNGState"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("rng_state")

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???
  def _typeCheck(a: Any): Boolean = ???

  def mkOrdering(sm: HailStateManager, missingEqual: Boolean)
    : is.hail.annotations.ExtendedOrdering = ???

  def scalaClassTag: scala.reflect.ClassTag[_ <: AnyRef] = ???

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
