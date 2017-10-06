package is.hail.utils

object UInt {
  def apply(i: Int): UInt = {
    assert(i >= 0, s"UInt cannot be constructed from negative integers. Found $i.")
    new UInt(i)
  }

  def apply(l: Long): UInt = {
    assert((l >>> 32) == 0, s"Long value does not fit in UInt. Found $l.")
    new UInt(l.toInt)
  }

  def uintFromRep(i: Int): UInt = new UInt(i)

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

  def compare(right: UInt): Int = {
    val l = toLong
    val r = right.toLong
    if (l == r) 0 else if (l > r) 1 else -1
  }

  def intRep: Int = i

  def +(right: UInt, dummy: Boolean = false): UInt = UInt(toLong + right.toLong)
  def +(right: Int): UInt = UInt(toLong + right)
  def +(right: Double): Double = toDouble + right
  def +(right: Long): Long = toLong + right

  def -(right: UInt, dummy: Boolean = false): UInt = UInt(toLong - right.toLong)
  def -(right: Int): UInt = UInt(toLong - right)
  def -(right: Double): Double = toDouble - right
  def -(right: Long): Long = toLong - right

  def *(right: UInt, dummy: Boolean = false): UInt = UInt(toLong * right.toLong)
  def *(right: Int): UInt = UInt(toLong * right)
  def *(right: Double): Double = toDouble * right
  def *(right: Long): Long = toLong * right

  def /(right: UInt, dummy: Boolean = false): UInt = UInt(toLong / right.toLong)
  def /(right: Int): UInt = UInt(toLong / right)
  def /(right: Double): Double = toDouble / right
  def /(right: Long): Long = toLong / right

  def <=(right: UInt, dummy: Boolean = false): Boolean = toLong <= right.toLong
  def <=(right: Int): Boolean = toLong <= right
  def <=(right: Double): Boolean = toDouble <= right
  def <=(right: Long): Boolean = toLong <= right

  def >=(right: UInt, dummy: Boolean = false): Boolean = toLong >= right.toLong
  def >=(right: Int): Boolean = toLong >= right
  def >=(right: Double): Boolean = toDouble >= right
  def >=(right: Long): Boolean = toLong >= right

  def <(right: UInt, dummy: Boolean = false): Boolean = toLong < right.toLong
  def <(right: Int): Boolean = toLong < right
  def <(right: Double): Boolean = toDouble < right
  def <(right: Long): Boolean = toLong < right

  def >(right: UInt, dummy: Boolean = false): Boolean = toLong > right.toLong
  def >(right: Int): Boolean = toLong > right
  def >(right: Double): Boolean = toDouble > right
  def >(right: Long): Boolean = toLong > right

  def ==(right: UInt, dummy: Boolean = false): Boolean = toLong == right.toLong
  def ==(right: Int): Boolean = toLong == right
  def ==(right: Double): Boolean = toDouble == right
  def ==(right: Long): Boolean = toLong == right

  def !=(right: UInt, dummy: Boolean = false): Boolean = toLong != right.toLong
  def !=(right: Int): Boolean = toLong != right
  def !=(right: Double): Boolean = toDouble != right
  def !=(right: Long): Boolean = toLong != right

  def toInt: Int = {
    assert(i >= 0)
    i
  }

  def toLong: Long = i & 0xffffffffL

  def toDouble: Double = toLong.toDouble

  def toFloat: Float = toLong.toFloat

  override def toString: String = toLong.toString
}

object ArrayUInt {
  def apply(n: Int): ArrayUInt = {
    val a = new Array[Int](n)
    new ArrayUInt(a)
  }
}

class ArrayUInt(val a: Array[Int]) extends AnyVal {
  def apply(i: Int): UInt = UInt.uintFromRep(a(i))

  def update(i: Int, x: UInt) {
    a(i) = x.intRep
  }

  def length: Int = a.length

  def sum: UInt = {
    var i = 0
    var s = UInt(0)
    while (i < length) {
      s += apply(i)
      i += 1
    }
    s
  }

  def intArrayRep: Array[Int] = a
}