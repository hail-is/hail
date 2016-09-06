package org.broadinstitute.hail.utils.richUtils

class RichBoolean(val b: Boolean) extends AnyVal {
  def ==>(that: => Boolean): Boolean = !b || that

  def iff(that: Boolean): Boolean = b == that

  def toInt: Double = if (b) 1 else 0

  def toDouble: Double = if (b) 1d else 0d
}
