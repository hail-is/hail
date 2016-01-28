package org.broadinstitute.hail.methods

object Filter {
  // FIXME
  def keepThisAny(a: Any, keep: Boolean): Boolean =
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
