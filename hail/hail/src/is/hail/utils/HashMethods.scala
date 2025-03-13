package is.hail.utils

import org.apache.commons.math3.random.RandomDataGenerator

object UnivHash32 {
  def apply(outBits: Int, rand: RandomDataGenerator): UnivHash32 =
    new UnivHash32(outBits, rand.getRandomGenerator.nextInt() | 1)
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
  def apply(outBits: Int, rand: RandomDataGenerator): TwoIndepHash32 =
    new TwoIndepHash32(
      outBits,
      rand.getRandomGenerator.nextInt(),
      rand.getRandomGenerator.nextInt(),
    )
}

class TwoIndepHash32(outBits: Int, a: Long, b: Long) extends (Int => Int) {
  require(0 <= outBits && outBits <= 32)

  def shift: Int = 64 - outBits

  override def apply(key: Int): Int = ((a * key + b) >>> shift).toInt
}

object SimpleTabulationHash32 {
  def apply(rand: RandomDataGenerator): SimpleTabulationHash32 = {
    val poly = PolyHash(rand, 32)
    new SimpleTabulationHash32(poly.fillIntArray(256 * 4))
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
    val poly = PolyHash(rand, 32)
    new TwistedTabulationHash32(poly.fillLongArray(256 * 4))
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
  def apply(rand: RandomDataGenerator): FiveIndepTabulationHash32 = {
    val poly1 = PolyHash(rand, 32)
    val poly2 = PolyHash(rand, 32)
    new FiveIndepTabulationHash32(
      poly1.fillLongArray(256 * 4),
      poly2.fillIntArray(259 * 3),
    )
  }
}

// compare to Thorup and Zhang, "Tabulation-Based 5-Independent Hashing with Applications to Linear Probing and
// Second Moment Estimation", section A.7
class FiveIndepTabulationHash32(keyTable: Array[Long], derivedKeyTable: Array[Int])
    extends (Int => Int) {
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

object PolyHash {
  def apply(rand: RandomDataGenerator, degree: Int): PolyHash =
    new PolyHash(Array.fill(degree)(rand.getRandomGenerator.nextInt()))

  // Can be done by the PCLMULQDQ "carryless multiply" instruction on x86 processors post ~2010.
  // This would give a significant speed boost. Any way to do this from JVM?
  def polyMult(a: Int, b: Int): Long = {
    var result: Long = 0
    if (b != 0) {
      var aBuf: Int = a
      var bBuf: Long = b & 0xffffffffL
      while (aBuf != 0) {
        result ^= bBuf * (aBuf & 1)
        aBuf >>>= 1
        bBuf <<= 1
      }
    }
    result
  }

  // Reduces g modulo the irreducible polynomial x^32 + x^7 + x^3 + x^2 + 1
  // following the method described in Intel white paper
  // "Intel Carry-Less Multiplication Instruction and its Usage for Computing the GCM Mode"
  def reduce(g: Long): Int = {
    val high = (g >>> 32).toInt
    val low = g.toInt
    val a = high ^ (high >>> 25) ^ (high >>> 29) ^ (high >>> 30)
    low ^ a ^ (a << 2) ^ (a << 3) ^ (a << 7)
  }

  def multGF(a: Int, b: Int): Int = reduce(polyMult(a, b))
}

class PolyHash(val coeffs: Array[Int]) extends (Int => Int) {
  import PolyHash._

  // polynomial evaluation using Horner's rule
  override def apply(x: Int): Int = {
    val deg = coeffs.length - 1
    var acc = 0
    var i = deg
    while (i >= 0) {
      acc = multGF(acc, x) ^ coeffs(i)
      i -= 1
    }
    acc
  }

  def fillIntArray(size: Int): Array[Int] =
    Array.tabulate(size)(apply)

  def fillLongArray(size: Int): Array[Long] =
    Array.tabulate(size)(i => (apply(i << 1).toLong << 32) | (apply((i << 1) | 1) & 0xffffffffL))
}
