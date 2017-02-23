package is.hail.methods

object Filter {
  def boxedKeepThis(a: java.lang.Boolean, keep: Boolean): Boolean =
    if (a == null)
      false
    else
      keepThis(a, keep)

  def keepThis(b: Boolean, keep: Boolean): Boolean =
    if (keep)
      b
    else
      !b
}
