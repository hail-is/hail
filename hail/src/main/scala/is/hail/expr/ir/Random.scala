package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.types.physical.stypes.concrete.SRNGState
import is.hail.utils.FastIndexedSeq
import net.sourceforge.jdistlib.rng.RandomEngine

object Threefry {
  val keyConst = 0x1BD11BDAA9FC1A22L

  val rotConsts = Array(
    Array(14, 16),
    Array(52, 57),
    Array(23, 40),
    Array( 5, 37),
    Array(25, 33),
    Array(46, 12),
    Array(58, 22),
    Array(32, 32))

  val defaultNumRounds = 20

  def expandKey(k: IndexedSeq[Long]): IndexedSeq[Long] = {
    assert(k.length == 4)
    val k4 = k(0) ^ k(1) ^ k(2) ^ k(3) ^ keyConst
    k :+ k4
  }

  def rotL(i: Value[Long], n: Value[Int]): Code[Long] = {
    (i << n) | (i >>> -n)
  }

  def mix(cb: CodeBuilderLike, x0: Settable[Long], x1: Settable[Long], n: Int): Unit = {
    cb.assign(x0, x0 + x1)
    cb.assign(x1, rotL(x1, n))
    cb.assign(x1, x0 ^ x1)
  }

  def injectKey(key: IndexedSeq[Long], tweak: Long, block: Array[Long], s: Int): Unit = {
    val tweakExt = Array[Long](tweak, 0, tweak)
    block(0) += key(s % 5)
    block(1) += key((s + 1) % 5) + tweakExt(s % 3)
    block(2) += key((s + 2) % 5) + tweakExt((s + 1) % 3)
    block(3) += key((s + 3) % 5) + s.toLong
  }

  def injectKey(cb: CodeBuilderLike,
    key: IndexedSeq[Long],
    tweak: Value[Long],
    block: IndexedSeq[Settable[Long]],
    s: Int
  ): Unit = {
    val tweakExt = Array[Value[Long]](tweak, const(0), tweak)
    cb.assign(block(0), block(0) + key(s % 5))
    cb.assign(block(1), block(1) + const(key((s + 1) % 5)) + tweakExt(s % 3))
    cb.assign(block(2), block(2) + const(key((s + 2) % 5)) + tweakExt((s + 1) % 3))
    cb.assign(block(3), block(3) + const(key((s + 3) % 5)) + const(s.toLong))
  }

  def permute(x: Array[Settable[Long]]): Unit = {
    val tmp = x(1)
    x(1) = x(3)
    x(3) = tmp
  }

  def encryptUnrolled(k0: Long, k1: Long, k2: Long, k3: Long, t: Long, _x0: Long, _x1: Long, _x2: Long, _x3: Long): Unit = {
    import java.lang.Long.rotateLeft
    var x0 = _x0
    var x1 = _x1
    var x2 = _x2
    var x3 = _x3
    val k4 = k0 ^ k1 ^ k2 ^ k3 ^ keyConst
    // d = 0
    // injectKey s = 0
    x0 += k0; x1 += k1 + t; x2 += k2; x3 += k3
    x0 += x1; x1 = rotateLeft(x1, 14); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 16); x3 ^= x2
    // d = 1
    x0 += x3; x3 = rotateLeft(x3, 52); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 57); x1 ^= x2
    // d = 2
    x0 += x1; x1 = rotateLeft(x1, 23); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 40); x3 ^= x2
    // d = 3
    x0 += x3; x3 = rotateLeft(x3, 5); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 37); x1 ^= x2
    // d = 4
    // injectKey s = 1
    x0 += k1; x1 += k2; x2 += k3 + t; x3 += k4 + 1
    x0 += x1; x1 = rotateLeft(x1, 25); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 33); x3 ^= x2
    // d = 5
    x0 += x3; x3 = rotateLeft(x3, 46); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 12); x1 ^= x2
    // d = 6
    x0 += x1; x1 = rotateLeft(x1, 58); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 22); x3 ^= x2
    // d = 7
    x0 += x3; x3 = rotateLeft(x3, 32); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 32); x1 ^= x2
    // d = 8
    // injectKey s = 2
    x0 += k2; x1 += k3 + t; x2 += k4 + t; x3 += k0 + 2
    x0 += x1; x1 = rotateLeft(x1, 14); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 16); x3 ^= x2
    // d = 9
    x0 += x3; x3 = rotateLeft(x3, 52); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 57); x1 ^= x2
    // d = 10
    x0 += x1; x1 = rotateLeft(x1, 23); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 40); x3 ^= x2
    // d = 11
    x0 += x3; x3 = rotateLeft(x3, 5); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 37); x1 ^= x2
    // d = 12
    // injectKey s = 3
    x0 += k3; x1 += k4 + t; x2 += k0; x3 += k1 + 3
    x0 += x1; x1 = rotateLeft(x1, 25); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 33); x3 ^= x2
    // d = 13
    x0 += x3; x3 = rotateLeft(x3, 46); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 12); x1 ^= x2
    // d = 14
    x0 += x1; x1 = rotateLeft(x1, 58); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 22); x3 ^= x2
    // d = 15
    x0 += x3; x3 = rotateLeft(x3, 32); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 32); x1 ^= x2
    // d = 16
    // injectKey s = 4
    x0 += k4; x1 += k0; x2 += k1 + t; x3 += k2 + 4
    x0 += x1; x1 = rotateLeft(x1, 14); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 16); x3 ^= x2
    // d = 17
    x0 += x3; x3 = rotateLeft(x3, 52); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 57); x1 ^= x2
    // d = 18
    x0 += x1; x1 = rotateLeft(x1, 23); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 40); x3 ^= x2
    // d = 19
    x0 += x3; x3 = rotateLeft(x3, 5); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 37); x1 ^= x2
    // d = 20
    // injectKey s = 5
    x0 += k0; x1 += k1 + t; x2 += k2 + t; x3 += k3 + 5
  }

  def encrypt(k: IndexedSeq[Long], t: Long, x: Array[Long]): Unit =
    encrypt(k, t, x, defaultNumRounds)

  def encrypt(k: IndexedSeq[Long], t: Long, x: Array[Long], rounds: Int): Unit = {
    assert(k.length == 5)
    assert(x.length == 4)

    for (d <- 0 until rounds) {
      if (d % 4 == 0)
        injectKey(k, t, x, d / 4)

      x(0) += x(1)
      x(1) = java.lang.Long.rotateLeft(x(1), rotConsts(d % 8)(0))
      x(1) ^= x(0)
      x(2) += x(3)
      x(3) = java.lang.Long.rotateLeft(x(3), rotConsts(d % 8)(1))
      x(3) ^= x(2)

      val tmp = x(1)
      x(1) = x(3)
      x(3) = tmp
    }

    if (rounds % 4 == 0)
      injectKey(k, t, x, rounds / 4)
  }

  def encrypt(cb: CodeBuilderLike,
    k: IndexedSeq[Long],
    t: Value[Long],
    x: IndexedSeq[Settable[Long]]
  ): Unit =
    encrypt(cb, k, t, x, defaultNumRounds)

  def encrypt(cb: CodeBuilderLike,
    k: IndexedSeq[Long],
    t: Value[Long],
    _x: IndexedSeq[Settable[Long]],
    rounds: Int
  ): Unit = {
    assert(k.length == 5)
    assert(_x.length == 4)
    val x = _x.toArray

    for (d <- 0 until rounds) {
      if (d % 4 == 0)
        injectKey(cb, k, t, x, d / 4)

      for (j <- 0 until 2)
        mix(cb, x(2*j), x(2*j+1), rotConsts(d % 8)(j))

      permute(x)
    }

    if (rounds % 4 == 0)
      injectKey(cb, k, t, x, rounds / 4)
  }

  def debugPrint(cb: EmitCodeBuilder, x: IndexedSeq[Settable[Long]], info: String) {
    cb.println(s"[$info]=\n\t", x(0).toString, "  ", x(1).toString, "  ", x(2).toString, "  ", x(3).toString)
  }

  def apply(k: IndexedSeq[Long]): AsmFunction2[Array[Long], Long, Unit] = {
    val f = FunctionBuilder[Array[Long], Long, Unit]("Threefry")
    f.mb.emitWithBuilder { cb =>
      val xArray = f.mb.getArg[Array[Long]](1)
      val t = f.mb.getArg[Long](2)
      val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"x$i", xArray(i)))
      encrypt(cb, expandKey(k), t, x)
      for (i <- 0 until 4) cb += (xArray(i) = x(i))
      Code._empty
    }
    f.result(false)(new HailClassLoader(getClass.getClassLoader))
  }
}

class RNGState {
  val staticAcc: Array[Long] = Array.fill(4)(0)
  val staticIdx: Int = 0
  val staticOpen: Array[Long] = Array.fill(4)(0)
  val staticOpenLen: Int = 0
  val dynAcc: Array[Long] = Array.fill(4)(0)
  val dynIdx: Int = 0
  val dynOpen: Array[Long] = Array.fill(4)(0)
  val dynOpenLen: Int = 0
}

object ThreefryRandomEngine {
  def apply(
    k1: Long, k2: Long, k3: Long, k4: Long,
    h1: Long, h2: Long, h3: Long, h4: Long,
    x1: Long, x2: Long, x3: Long
  ): ThreefryRandomEngine = {
    new ThreefryRandomEngine(
      Threefry.expandKey(FastIndexedSeq(k1, k2, k3, k4)),
      Array(h1 ^ x1, h2 ^ x2, h3 ^ x3, h4),
      0)
  }

  def apply(): ThreefryRandomEngine = {
    val rand = new java.util.Random()
    new ThreefryRandomEngine(
      Threefry.expandKey(Array.fill(4)(rand.nextLong())),
      Array.fill(4)(rand.nextLong()),
      0)
  }
}

class ThreefryRandomEngine(
  val key: IndexedSeq[Long],
  val state: Array[Long],
  var counter: Long,
  val tweak: Long = SRNGState.finalBlockNoPadTweak
) extends RandomEngine {
  val buffer: Array[Long] = Array.ofDim[Long](4)
  var usedInts: Int = 8
  var hasBufferedGaussian: Boolean = false
  var bufferedGaussian: Double = 0.0

  override def clone(): ThreefryRandomEngine = ???

  private def fillBuffer(): Unit = {
    import java.lang.Long.rotateLeft
    var x0 = state(0)
    var x1 = state(1)
    var x2 = state(2)
    var x3 = state(3) ^ counter
    val k0 = key(0); val k1 = key(1); val k2 = key(2); val k3 = key(3)
    val k4 = k0 ^ k1 ^ k2 ^ k3 ^ Threefry.keyConst
    val t = tweak
    // d = 0
    // injectKey s = 0
    x0 += k0; x1 += k1 + t; x2 += k2; x3 += k3
    x0 += x1; x1 = rotateLeft(x1, 14); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 16); x3 ^= x2
    // d = 1
    x0 += x3; x3 = rotateLeft(x3, 52); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 57); x1 ^= x2
    // d = 2
    x0 += x1; x1 = rotateLeft(x1, 23); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 40); x3 ^= x2
    // d = 3
    x0 += x3; x3 = rotateLeft(x3, 5); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 37); x1 ^= x2
    // d = 4
    // injectKey s = 1
    x0 += k1; x1 += k2; x2 += k3 + t; x3 += k4 + 1
    x0 += x1; x1 = rotateLeft(x1, 25); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 33); x3 ^= x2
    // d = 5
    x0 += x3; x3 = rotateLeft(x3, 46); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 12); x1 ^= x2
    // d = 6
    x0 += x1; x1 = rotateLeft(x1, 58); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 22); x3 ^= x2
    // d = 7
    x0 += x3; x3 = rotateLeft(x3, 32); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 32); x1 ^= x2
    // d = 8
    // injectKey s = 2
    x0 += k2; x1 += k3 + t; x2 += k4 + t; x3 += k0 + 2
    x0 += x1; x1 = rotateLeft(x1, 14); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 16); x3 ^= x2
    // d = 9
    x0 += x3; x3 = rotateLeft(x3, 52); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 57); x1 ^= x2
    // d = 10
    x0 += x1; x1 = rotateLeft(x1, 23); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 40); x3 ^= x2
    // d = 11
    x0 += x3; x3 = rotateLeft(x3, 5); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 37); x1 ^= x2
    // d = 12
    // injectKey s = 3
    x0 += k3; x1 += k4 + t; x2 += k0; x3 += k1 + 3
    x0 += x1; x1 = rotateLeft(x1, 25); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 33); x3 ^= x2
    // d = 13
    x0 += x3; x3 = rotateLeft(x3, 46); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 12); x1 ^= x2
    // d = 14
    x0 += x1; x1 = rotateLeft(x1, 58); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 22); x3 ^= x2
    // d = 15
    x0 += x3; x3 = rotateLeft(x3, 32); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 32); x1 ^= x2
    // d = 16
    // injectKey s = 4
    x0 += k4; x1 += k0; x2 += k1 + t; x3 += k2 + 4
    x0 += x1; x1 = rotateLeft(x1, 14); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 16); x3 ^= x2
    // d = 17
    x0 += x3; x3 = rotateLeft(x3, 52); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 57); x1 ^= x2
    // d = 18
    x0 += x1; x1 = rotateLeft(x1, 23); x1 ^= x0
    x2 += x3; x3 = rotateLeft(x3, 40); x3 ^= x2
    // d = 19
    x0 += x3; x3 = rotateLeft(x3, 5); x3 ^= x0
    x2 += x1; x1 = rotateLeft(x1, 37); x1 ^= x2
    // d = 20
    // injectKey s = 5
    x0 += k0; x1 += k1 + t; x2 += k2 + t; x3 += k3 + 5

    buffer(0) = x0; buffer(1) = x1; buffer(2) = x2; buffer(3) = x3
    counter += 1
    usedInts = 0
  }

  override def setSeed(seed: Long): Unit = ???

  override def getSeed: Long = ???

  override def nextLong(): Long = {
    usedInts += usedInts & 1 // round up to multiple of 2
    if (usedInts >= 8) fillBuffer()
    val result = buffer(usedInts >> 1)
    usedInts += 2
    result
  }

  override def nextInt(): Int = {
    if (usedInts >= 8) fillBuffer()
    val result = buffer(usedInts >> 1)
    usedInts += 1
    val parity = usedInts & 1
    val shift = parity << 5 // either 0 or 32
    (result >>> shift).toInt // either first or second 32 bits
  }

  // Uses approach from https://github.com/apple/swift/pull/39143
  override def nextInt(n: Int): Int = {
    val nL = n.toLong
    val mult = nL * (nextInt().toLong & 0xFFFFFFFFL)
    val result = (mult >>> 32).toInt
    val fraction = mult & 0xFFFFFFFFL

    // optional early return, benchmark to decide if it helps
    if (fraction < ((1L << 32) - nL)) return result

    val multHigh = (((nL * (nextInt().toLong & 0xFFFFFFFFL)) >>> 32) + (nL * (nextInt().toLong & 0xFFFFFFFFL))) >>> 32
    val sum = fraction + multHigh
    val carry = (sum >>> 32).toInt
    result + carry
  }

  // Uses standard Java approach. We could use the same approach as for ints,
  // but that requires full-width multiplication of two longs, which adds some
  // complexity.
  override def nextLong(l: Long): Long = {
    var x = nextLong() >>> 1
    var r = x % l
    while (x - r + (l - 1) < 0) {
      x = nextLong() >>> 1
      r = x % l
    }
    r
  }

  override def nextGaussian(): Double = {
    if (hasBufferedGaussian) {
      hasBufferedGaussian = false
      return bufferedGaussian
    }

    var v1 = 2 * nextDouble() - 1 // between -1 and 1
    var v2 = 2 * nextDouble() - 1
    var s = v1 * v1 + v2 * v2
    while (s >= 1 || s == 0) {
      v1 = 2 * nextDouble() - 1 // between -1 and 1
      v2 = 2 * nextDouble() - 1
      s = v1 * v1 + v2 * v2
    }
    val multiplier = StrictMath.sqrt(-2 * StrictMath.log(s) / s)
    bufferedGaussian = v2 * multiplier
    hasBufferedGaussian = true
    v1 * multiplier
  }

  // Equivalent to generating an infinite-precision real number in [0, 1),
  // represented as an infinitely long bitstream, and rounding down to the
  // nearest representable floating point number.
  // In contrast, the standard Java and jdistlib generators sample uniformly
  // from a sequence of equidistant floating point numbers in [0, 1), using
  // (nextLong() >>> 11).toDouble / (1L << 53)
  //
  // Intuitively, the algorithm is:
  // * lazily generate an infinite string of random bits, interpreted as
  //   the binary expansion of a real number in [0, 1), i.e. `0.${bits}`
  // * convert to floating point representation: the exponent is -n, where n is
  //   the number of 0s before the first 1, and the significand is the first 1
  //   followed by the next 52 bits.
  override def nextDouble(): Double = {
    // first generate random bits until we get the first 1, counting the number
    // of zeroes
    var bits: Long = nextLong()
    // the exponent starts at 1022 and subtracts the number of leading zeroes,
    // to account for the exponent bias in IEE754
    var exponent: Int = 1022
    while (bits == 0) {
      bits = nextLong()
      exponent -= 64
    }
    // use trailing zeroes instead of leading zeroes as slight optimization,
    // but probabilistically equivalent
    val e = java.lang.Long.numberOfTrailingZeros(bits)
    exponent -= e
    // If there are at least 52 bits before the trailing 1, use those
    val significand = (if (e < 12) bits else nextLong()) >>> 12
    val result = (exponent.toLong << 52) | significand
    java.lang.Double.longBitsToDouble(result)
  }

  override def nextFloat(): Float = {
    // first generate random bits until we get the first 1, counting the number
    // of zeroes
    var bits: Int = nextInt()
    // the exponent starts at 126 and subtracts the number of leading zeroes,
    // to account for the exponent bias in IEE754
    var exponent: Int = 126
    while (bits == 0) {
      bits = nextInt()
      exponent -= 32
    }
    // use trailing zeroes instead of leading zeroes as slight optimization,
    // but probabilistically equivalent
    val e = java.lang.Long.numberOfTrailingZeros(bits)
    exponent -= e
    // If there are at least 23 bits before the trailing 1, use those
    val significand = (if (e < 9) bits else nextInt()) >>> 9
    val result = (exponent << 23) | significand
    java.lang.Float.intBitsToFloat(result)
  }
}