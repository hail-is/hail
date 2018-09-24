package is.hail.utils.richUtils

class RichBoolean(val b: Boolean) extends AnyVal {
  def ==>(that: => Boolean): Boolean = !b || that

  def iff(that: Boolean): Boolean = b == that

  def toByte: Byte = if (b) 1 else 0

  def toInt: Int = if (b) 1 else 0

  def toLong: Long = if (b) 1L else 0L

  def toFloat: Float = if (b) 1.0f else 0.0f

  def toDouble: Double = if (b) 1d else 0d
}
