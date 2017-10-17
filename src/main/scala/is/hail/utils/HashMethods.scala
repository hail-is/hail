package is.hail.utils

import org.apache.commons.math3.random.RandomDataGenerator

object UnivHash32 {
  def apply(outBits: Int, rand: RandomDataGenerator): UnivHash32 = {
    new UnivHash32(outBits, rand.getRandomGenerator.nextInt() | 1)
  }
}

// see e.g. Thorup, "High Speed Hashing for Integers and Strings", for explanation of "multiply-shift" hash functions,
// both the universal and 2-independent constructions
class UnivHash32(outBits: Int, factor: Int) extends (Int => Int) {
  require(0 <= outBits && outBits <= 32)
  require((factor & 1) == 1)

  def shift: Int = 32 - outBits

  override def apply(key: Int): Int = (key * factor) >>> shift
}

object TwoIndepHash32 {
  def apply(outBits: Int, rand: RandomDataGenerator): TwoIndepHash32 = {
    new TwoIndepHash32(outBits, rand.getRandomGenerator.nextInt(), rand.getRandomGenerator.nextInt())
  }
}

class TwoIndepHash32(outBits: Int, a: Long, b: Long) extends (Int => Int) {
  require(0 <= outBits && outBits <= 32)

  def shift: Int = 64 - outBits

  override def apply(key: Int): Int = ((a * key + b) >>> shift).toInt
}
