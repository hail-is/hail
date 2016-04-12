package org.broadinstitute.hail.methods

object Filter {
  def keepThis(a: Option[Boolean], keep: Boolean): Boolean = a.map(x => keepThis(x, keep)).getOrElse(false)

  def keepThis(b: Boolean, keep: Boolean): Boolean =
    if (keep)
      b
    else
      !b
}
