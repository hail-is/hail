package is.hail.utils

// see e.g. Thorup, "High Speed Hashing for Integers and Strings", for explanation of "multiply-shift" hash functions,
// both the universal and 2-independent constructions
class UnivHash32(outBits: Int, factor: Int) extends (Int => Int) {
  require(0 <= outBits && outBits <= 32)

  val oddFactor = factor | 1
  val shift: Int = 32 - outBits

  override def apply(key: Int): Int = (key * oddFactor) >>> shift
}

class TwoIndepHash32(outBits: Int, a: Long, b: Long) extends (Int => Int) {
  require(0 <= outBits && outBits <= 32)

  val shift: Int = 64 - outBits

  override def apply(key: Int): Int = ((a * key + b) >>> shift).toInt
}
