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

object SimpleTabulationHash32 {
  def apply(rand: RandomDataGenerator): SimpleTabulationHash32 = {
    new SimpleTabulationHash32(Array.fill[Array[Int]](4)(Array.fill[Int](256)(rand.getRandomGenerator.nextInt())))
  }
}

// see e.g. Thorup, "Fast and Powerful Hashing using Tabulation", for explanation of both simple and twisted tabulation
// hashing
class SimpleTabulationHash32(table: Array[Array[Int]]) extends (Int => Int) {
  require(table.length == 4)
  require(table(0).length == 256)

  override def apply(key: Int): Int = {
    var out: Int = 0
    var keyBuffer: Int = key
    var i = 0
    while (i < 4) {
      out ^= table(i)(keyBuffer & 255)
      i += 1
      keyBuffer >>>= 8
    }
    out
  }
}

object TwistedTabulationHash32 {
  def apply(rand: RandomDataGenerator): TwistedTabulationHash32 = {
    new TwistedTabulationHash32(Array.fill[Array[Long]](4)(Array.fill[Long](256)(rand.getRandomGenerator.nextLong())))
  }
}

class TwistedTabulationHash32(table: Array[Array[Long]]) extends (Int => Int) {
  require(table.length == 4)
  require(table.forall(_.length == 256))

  override def apply(key: Int): Int = {
    var out: Long = 0
    var keyBuffer: Int = key
    var i = 0
    while (i < 3) {
      out ^= table(i)(keyBuffer & 255)
      i += 1
      keyBuffer >>>= 8
    }
    out ^= table(i)((keyBuffer ^ out.toInt) & 255)
    (out >>> 32).toInt
  }
}

object FiveIndepTabulationHash32 {
  def apply(rand: RandomDataGenerator): FiveIndepTabulationHash32 =
    new FiveIndepTabulationHash32(
      Array.fill[Long](256 * 4)(rand.getRandomGenerator.nextLong()),
      Array.fill[Int](259 * 3)(rand.getRandomGenerator.nextInt())
    )
}

// compare to Thorup and Zhang, "Tabulation-Based 5-Independent Hashing with Applications to Linear Probing and
// Second Moment Estimation", section A.7
class FiveIndepTabulationHash32(keyTable: Array[Long], derivedKeyTable: Array[Int]) extends (Int => Int) {
  require(keyTable.length == 256 * 4)
  require(derivedKeyTable.length == 259 * 3)

  override def apply(key: Int): Int = {
    var out: Int = 0
    var derivedKeys: Int = 0
    var keyBuffer: Int = key
    var offset = 0

    while (offset < 1024) {
      val compoundEntry: Long = keyTable(offset + (keyBuffer & 0xff))
      out ^= (compoundEntry & 0xffffffffL).toInt
      derivedKeys += (compoundEntry >>> 32).toInt
      offset += 256
      keyBuffer >>>= 8
    }

    def mask1: Int = 0x00300c03 // == (3 << 20) + (3 << 10) + 3
    def mask2: Int = 0x0ff3fcff // == (255 << 20) + (255 << 10) + 255
    derivedKeys = mask1 + (derivedKeys & mask2) - ((derivedKeys >> 8) & mask1)

    offset = 0
    while (offset < 3) {
      out ^= derivedKeyTable(offset + (derivedKeys & 0x3ff))
      offset += 261
      derivedKeys >>> 10
    }
    out
  }
}
