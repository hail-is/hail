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
    new SimpleTabulationHash32(Array.fill[Int](256 * 4)(rand.getRandomGenerator.nextInt()))
  }
}

// see e.g. Thorup, "Fast and Powerful Hashing using Tabulation", for explanation of both simple and twisted tabulation
// hashing
class SimpleTabulationHash32(table: Array[Int]) extends (Int => Int) {
  require(table.length == 256 * 4)

  override def apply(key: Int): Int = {
    var out: Int = 0
    var keyBuffer: Int = key
    var offset = 0
    while (offset < 1024) {
      out ^= table(offset + (keyBuffer & 0xff))
      offset += 256
      keyBuffer >>>= 8
    }
    out
  }
}

object TwistedTabulationHash32 {
  def apply(rand: RandomDataGenerator): TwistedTabulationHash32 = {
    new TwistedTabulationHash32(Array.fill[Long](256 * 4)(rand.getRandomGenerator.nextLong()))
  }
}

class TwistedTabulationHash32(table: Array[Long]) extends (Int => Int) {
  require(table.length == 256 * 4)

  override def apply(key: Int): Int = {
    var out: Long = 0
    var keyBuffer: Int = key
    var offset = 0
    while (offset < 768) {
      out ^= table(offset + (keyBuffer & 0xff))
      offset += 256
      keyBuffer >>>= 8
    }
    out ^= table(offset + ((keyBuffer ^ out.toInt) & 0xff))
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

class PolyHash {

  // Can be done by the PCLMULQDQ "carryless multiply" instruction on x86 processors post ~2010.
  // This would give a significant speed boost. Any way to do this from JVM?
  def polyMult(a: Int, b: Int): Long = {
    var result: Long = 0
    if (b != 0) {
      var aBuf: Int = a
      var bBuf: Long = b & 0xffffffffL
      while (aBuf != 0) {
        if ((aBuf & 1) == 1) result ^= bBuf
        aBuf >>>= 1
        bBuf <<= 1
      }
    }
    result
  }
}
