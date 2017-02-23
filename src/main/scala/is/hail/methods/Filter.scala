package is.hail.methods

object Filter {
  def keepThis(a: Any, keep: Boolean): Boolean =
    if (a == null)
      false
    else
      keepThis(a.asInstanceOf[Boolean], keep)

  def keepThis(b: Boolean, keep: Boolean): Boolean =
    if (keep)
      b
    else
      !b
}
