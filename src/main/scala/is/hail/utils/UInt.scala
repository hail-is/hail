package is.hail.utils

object UInt {
  def apply(i: Int): UInt = new UInt(i)

  def apply(l: Long): UInt = {
    assert((l >>> 32) == 0, s"Long value does not fit in UInt. Found $l.")
    UInt(l.toInt)
  }

  implicit val numeric: Numeric[UInt] = new Numeric[UInt] {
    override def plus(x: UInt, y: UInt): UInt = x + y

    override def minus(x: UInt, y: UInt): UInt = x - y

    override def times(x: UInt, y: UInt): UInt = x * y

    override def negate(x: UInt): UInt = throw new UnsupportedOperationException()

    override def fromInt(x: Int): UInt = UInt(x)

    override def toInt(x: UInt): Int = x.i

    override def toLong(x: UInt): Long = x.toLong

    override def toFloat(x: UInt): Float = x.toFloat

    override def toDouble(x: UInt): Double = x.toDouble

    override def compare(x: UInt, y: UInt): Int = x.compare(y)

    override def zero: UInt = UInt(0)

    override def one: UInt = UInt(1)
  }
}

class UInt private(val i: Int) extends AnyVal {
//  require(i >= 0)

  def compare(right: UInt): Int = {
    val l = toLong
    val r = right.toLong
    if (l == r) 0 else if (l > r) 1 else -1
  }

  def +(right: UInt): UInt = UInt(toLong + right.toLong)

  def -(right: UInt): UInt = UInt(toLong - right.toLong)

  def *(right: UInt): UInt = UInt(toLong * right.toLong)

  def /(right: UInt): UInt = UInt(toLong / right.toLong)

  def <=(right: UInt): Boolean = toLong <= right.toLong

  def >=(right: UInt): Boolean = toLong >= right.toLong

  def <(right: UInt): Boolean = toLong < right.toLong

  def >(right: UInt): Boolean = toLong > right.toLong

  def ==(right: UInt): Boolean = toLong == right.toLong

  def !=(right: UInt): Boolean = toLong != right.toLong

  def toInt: Int = {
    assert(i >= 0)
    i
  }

  def toLong: Long = i & 0xffffffffL

  def toDouble: Double = toLong.toDouble

  def toFloat: Float = toLong.toFloat

  override def toString: String = toLong.toString
}
