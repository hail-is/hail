package is.hail.utils

import java.security.SecureRandom

object UnivHash32 {
  def apply(outBits: Int = 32): UnivHash32 = {
    val random = new SecureRandom()
    new UnivHash32(outBits, random.nextInt() | 1)
  }
}

// see e.g. Thorup, "High Speed Hashing for Integers and Strings", for explanation of "multiply-shift" hash functions,
// both the universal and 2-independent constructions
class UnivHash32(outBits: Int, seed: Int) extends Function1[Int,Int] {
  require(0 <= outBits && outBits <= 32)
  require((seed & 1) == 1)
  val shift: Int = 32 - outBits
  override def apply(key: Int): Int = (key * seed) >>> shift
}

object TwoIndepHash32 {
  def apply(outBits: Int = 32): TwoIndepHash32 = {
    val random = new SecureRandom()
    new TwoIndepHash32(outBits, random.nextInt(), random.nextInt())
  }
}

class TwoIndepHash32(outBits: Int, a: Long, b: Long) extends Function1[Int,Int] {
  require(0 <= outBits && outBits <= 32)

  val shift: Int = 64 - outBits
  override def apply(key: Int): Int = ((a * key + b) >>> shift).toInt
}

object SimpleTabulationHash32 {
  def apply(): SimpleTabulationHash32 = {
    val random = new SecureRandom()
    new SimpleTabulationHash32(Array.fill[Array[Int]](4)(random.ints(256).toArray()))
  }
}

// see e.g. Thorup, "Fast and Powerful Hashing using Tabulation", for explanation of both simple and twisted tabulation
// hashing
class SimpleTabulationHash32(table: Array[Array[Int]]) extends Function1[Int,Int] {
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
  def apply(): TwistedTabulationHash32 = {
    val random = new SecureRandom()
    new TwistedTabulationHash32(Array.fill[Array[Long]](4)(random.longs(256).toArray()))
  }
}

class TwistedTabulationHash32(table: Array[Array[Long]]) extends Function1[Int,Int] {
  require(table.length == 4)
  for (i <- 0 to 3) require(table(i).length == 256)

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
  def apply(): FiveIndepTabulationHash32 = {
    val random = new SecureRandom()
    new FiveIndepTabulationHash32(
      Array.fill[Array[Long]](4)(random.longs(256).toArray()),
      Array.fill[Array[Int]](3)(random.ints(261).toArray())
    )
  }
}

// compare to Thorup and Zhang, "Tabulation-Based 5-Independent Hashing with Applications to Linear Probing and
// Second Moment Estimation", section A.7
class FiveIndepTabulationHash32(keyTable: Array[Array[Long]], derivedKeyTable: Array[Array[Int]]) extends Function1[Int,Int] {
  require(keyTable.length == 4)
  require(derivedKeyTable.length == 3)
  for (i <- 0 to 3) require(keyTable(i).length == 256)
  for (i <- 0 to 2) require(derivedKeyTable(i).length == 261)

  override def apply(key: Int): Int = {
    var out: Int = 0
    var derivedKeys: Int = 0
    var keyBuffer: Int = key
    var i = 0
    val mask32: Long = (1 << 32) - 1

    while (i < 4) {
      val compoundEntry: Long = keyTable(i)(keyBuffer & 255)
      out ^= (compoundEntry & mask32).toInt
      derivedKeys += (compoundEntry >>> 32).toInt
      i += 1
      keyBuffer >>>= 8
    }

    val mask1: Int = (3 << 20) + (3 << 10) + 3
    val mask2: Int = (255 << 20) + (255 << 10) + 255
    derivedKeys = mask1 + (derivedKeys & mask2) - ((derivedKeys >> 8) & mask1)

    i = 0
    while (i < 3) {
      out ^= derivedKeyTable(i)(derivedKeys & 1023)
      i += 1
      derivedKeys >>> 10
    }
    out
  }
}
