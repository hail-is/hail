package is.hail.types.virtual



case object TRNGState extends Type {
  override def _toPretty = "RNGState"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("rng_state")

  def _typeCheck(a: Any): Boolean = ???

  def mkOrdering(missingEqual: Boolean)
    : is.hail.annotations.ExtendedOrdering = ???

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
