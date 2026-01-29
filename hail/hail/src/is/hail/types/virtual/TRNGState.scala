package is.hail.types.virtual

import is.hail.backend.HailStateManager

case object TRNGState extends Type {
  override def _toPretty = "RNGState"

  override def pyString(sb: StringBuilder): Unit =
    sb ++= "rng_state"

  override def _typeCheck(a: Any): Boolean = ???

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean)
    : is.hail.annotations.ExtendedOrdering = ???

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
