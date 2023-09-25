package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.utils.FastSeq
import net.sourceforge.jdistlib.rng.RandomEngine
import net.sourceforge.jdistlib.{Beta, Gamma, HyperGeometric, Poisson}
import org.apache.commons.math3.random.RandomGenerator

object Threefry {
  val staticTweak = -1L
  val finalBlockNoPadTweak = -2L
  val finalBlockPaddedTweak = -3L

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

  val defaultKey: IndexedSeq[Long] =
    expandKey(FastSeq(0x215d6dfdb7dfdf6bL, 0x045cfa043329c49fL, 0x9ec75a93692444ddL, 0x1284681663220f1cL))

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

  def injectKey(key: IndexedSeq[Long], tweak: IndexedSeq[Long], block: Array[Long], s: Int): Unit = {
    assert(tweak.length == 3)
    assert(key.length == 5)
    assert(block.length == 4)
    block(0) += key(s % 5)
    block(1) += key((s + 1) % 5) + tweak(s % 3)
    block(2) += key((s + 2) % 5) + tweak((s + 1) % 3)
    block(3) += key((s + 3) % 5) + s.toLong
  }

  def injectKey(cb: CodeBuilderLike,
    key: IndexedSeq[Long],
    tweak: IndexedSeq[Value[Long]],
    block: IndexedSeq[Settable[Long]],
    s: Int
  ): Unit = {
    cb.assign(block(0), block(0) + key(s % 5))
    cb.assign(block(1), block(1) + const(key((s + 1) % 5)) + tweak(s % 3))
    cb.assign(block(2), block(2) + const(key((s + 2) % 5)) + tweak((s + 1) % 3))
    cb.assign(block(3), block(3) + const(key((s + 3) % 5)) + const(s.toLong))
  }

  def permute(x: Array[Settable[Long]]): Unit = {
    val tmp = x(1)
    x(1) = x(3)
    x(3) = tmp
  }

  def encryptUnrolled(k0: Long, k1: Long, k2: Long, k3: Long, k4: Long, t0: Long, t1: Long, x: Array[Long]): Unit = {
    import java.lang.Long.rotateLeft
    var x0 = x(0); var x1 = x(1); var x2 = x(2); var x3 = x(3)
    val t2 = t0 ^ t1

    // d = 0
    // injectKey s = 0
    x0 += k0; x1 += k1 + t0; x2 += k2 + t1; x3 += k3
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
    x0 += k1; x1 += k2 + t1; x2 += k3 + t2; x3 += k4 + 1
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
    x0 += k2; x1 += k3 + t2; x2 += k4 + t0; x3 += k0 + 2
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
    x0 += k3; x1 += k4 + t0; x2 += k0 + t1; x3 += k1 + 3
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
    x0 += k4; x1 += k0 + t1; x2 += k1 + t2; x3 += k2 + 4
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
    x0 += k0; x1 += k1 + t2; x2 += k2 + t0; x3 += k3 + 5

    x(0) = x0; x(1) = x1; x(2) = x2; x(3) = x3
  }

  def encrypt(k: IndexedSeq[Long], t: IndexedSeq[Long], x: Array[Long]): Unit =
    encrypt(k, t, x, defaultNumRounds)

  def encrypt(k: IndexedSeq[Long], _t: IndexedSeq[Long], x: Array[Long], rounds: Int): Unit = {
    assert(k.length == 5)
    assert(_t.length == 2)
    assert(x.length == 4)
    val t = Array(_t(0), _t(1), _t(0) ^ _t(1))

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
    t: IndexedSeq[Value[Long]],
    x: IndexedSeq[Settable[Long]]
  ): Unit =
    encrypt(cb, k, t, x, defaultNumRounds)

  def encrypt(cb: CodeBuilderLike,
    k: IndexedSeq[Long],
    _t: IndexedSeq[Value[Long]],
    _x: IndexedSeq[Settable[Long]],
    rounds: Int
  ): Unit = {
    assert(k.length == 5)
    assert(_t.length == 2)
    assert(_x.length == 4)
    val x = _x.toArray
    val t = Array(_t(0), _t(1), cb.memoize(_t(0) ^ _t(1)))

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

  def pmac(sum: Array[Long], message: IndexedSeq[Long]): Array[Long] = {
    val (hash, finalTweak) = pmacHashFromState(sum, message)
    encrypt(Threefry.defaultKey, Array(finalTweak, 0L), hash)
    hash
  }

  def pmac(nonce: Long, staticID: Long, message: IndexedSeq[Long]): Array[Long] = {
    val sum = Array(nonce, staticID, 0L, 0L)
    encrypt(Threefry.defaultKey, Array(Threefry.staticTweak, 0L), sum)
    pmac(sum, message)
  }

  def pmac(message: IndexedSeq[Long]): Array[Long] = {
    val sum = Array.ofDim[Long](4)
    pmac(sum, message)
  }

  def pmacHash(nonce: Long, staticID: Long, message: IndexedSeq[Long]): (Array[Long], Long) = {
    val sum = Array(nonce, staticID, 0L, 0L)
    encrypt(Threefry.defaultKey, Array(Threefry.staticTweak, 0L), sum)
    pmacHashFromState(sum, message)
  }

  def pmacHashFromState(sum: Array[Long], _message: IndexedSeq[Long]): (Array[Long], Long) = {
    val length = _message.length
    val paddedLength = Math.max((length + 3) & (~3), 4)
    val padded = (paddedLength != length)
    val message = Array.ofDim[Long](paddedLength)
    _message.copyToArray(message)
    if (padded) message(length) = 1L

    var i = 0
    while (i + 4 < paddedLength) {
      val x = message.slice(i, i + 4)
      encrypt(Threefry.defaultKey, Array(i.toLong, 0L), x)
      sum(0) ^= x(0)
      sum(1) ^= x(1)
      sum(2) ^= x(2)
      sum(3) ^= x(3)
      i += 4
    }
    for (j <- 0 until 4) {
      sum(j) ^= message(i + j)
    }
    val finalTweak = if (padded) Threefry.finalBlockPaddedTweak else Threefry.finalBlockNoPadTweak

    (sum, finalTweak)
  }

  def apply(k: IndexedSeq[Long]): AsmFunction2[Array[Long], Array[Long], Unit] = {
    val f = FunctionBuilder[Array[Long], Array[Long], Unit]("Threefry")
    f.mb.emitWithBuilder { cb =>
      val xArray = f.mb.getArg[Array[Long]](1)
      val tArray = f.mb.getArg[Array[Long]](2)
      val t = Array(cb.memoize(tArray(0)), cb.memoize(tArray(1)))
      val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"x$i", xArray(i)))
      encrypt(cb, expandKey(k), t, x)
      for (i <- 0 until 4) cb += (xArray(i) = x(i))
      Code._empty
    }
    f.result(false)(new HailClassLoader(getClass.getClassLoader))
  }
}

class PMacHash() {
  val sum = Array.ofDim[Long](4)
  var i = 0
  val buffer = Array.ofDim[Long](4)
  var curOffset = 0

  def extend(a: Array[Long]): PMacHash = {
    val n = a.length
    var j = 0
    while (4 - curOffset < n - j) {
      val lenCopied = 4 - curOffset
      Array.copy(a, j, buffer, curOffset, lenCopied)
      Threefry.encrypt(Threefry.defaultKey, Array(i.toLong, 0L), buffer)
      sum(0) ^= buffer(0)
      sum(1) ^= buffer(1)
      sum(2) ^= buffer(2)
      sum(3) ^= buffer(3)
      curOffset = 0
      j += lenCopied
      i += 1
    }
    Array.copy(a, j, buffer, curOffset, n - j)
    curOffset += n - j
    this
  }

  def hash: Array[Long] = {
    assert(i == 0 || curOffset > 0)
    val finalTweak = if (curOffset < 4) {
      buffer(curOffset) = 1
      curOffset += 1
      Threefry.finalBlockPaddedTweak
    } else
      Threefry.finalBlockNoPadTweak
    var j = 0
    while (j < curOffset) {
      sum(j) ^= buffer(j)
      j += 1
    }
    Threefry.encrypt(Threefry.defaultKey, Array(finalTweak, 0L), sum)
    sum
  }
}

object ThreefryRandomEngine {
  def apply(): ThreefryRandomEngine = {
    val key = Threefry.defaultKey
    new ThreefryRandomEngine(
      key(0), key(1), key(2), key(3), key(4),
      0, 0, 0, 0, 0, 0)
  }

  def apply(nonce: Long, staticID: Long, message: IndexedSeq[Long]): ThreefryRandomEngine = {
    val engine = ThreefryRandomEngine()
    val (hash, finalTweak) = Threefry.pmacHash(nonce, staticID, message)
    engine.resetState(hash(0), hash(1), hash(2), hash(3), finalTweak)
    engine
  }

  def randState(): ThreefryRandomEngine = {
    val rand = new java.util.Random()
    val key = Threefry.expandKey(Array.fill(4)(rand.nextLong()))
    new ThreefryRandomEngine(
      key(0), key(1), key(2), key(3), key(4),
      rand.nextLong(), rand.nextLong(), rand.nextLong(), rand.nextLong(),
      0, 0)
  }
}

class ThreefryRandomEngine(
  val k0: Long,
  val k1: Long,
  val k2: Long,
  val k3: Long,
  val k4: Long,
  var state0: Long,
  var state1: Long,
  var state2: Long,
  var state3: Long,
  var counter: Long,
  var tweak: Long
) extends RandomEngine with RandomGenerator {
  val buffer: Array[Long] = Array.ofDim[Long](4)
  var usedInts: Int = 8
  var hasBufferedGaussian: Boolean = false
  var bufferedGaussian: Double = 0.0

  override def clone(): ThreefryRandomEngine = ???

  def resetState(s0: Long, s1: Long, s2: Long, s3: Long, _tweak: Long): Unit = {
    state0 = s0
    state1 = s1
    state2 = s2
    state3 = s3
    tweak = _tweak
    counter = 0
    usedInts = 8
    hasBufferedGaussian = false
  }

  private[this] val poisState = Poisson.create_random_state()

  def runif(min: Double, max: Double): Double = min + (max - min) * nextDouble()

  def rnorm(mean: Double, sd: Double): Double = mean + sd * nextGaussian()

  def rpois(lambda: Double): Double = Poisson.random(lambda, this, poisState)

  def rbeta(a: Double, b: Double): Double = Beta.random(a, b, this)

  def rgamma(shape: Double, scale: Double): Double = Gamma.random(shape, scale, this)

  def rhyper(numSuccessStates: Double, numFailureStates: Double, numToDraw: Double): Double =
    HyperGeometric.random(numSuccessStates, numFailureStates, numToDraw, this)

  private def fillBuffer(): Unit = {
    buffer(0) = state0; buffer(1) = state1; buffer(2) = state2; buffer(3) = state3
    Threefry.encryptUnrolled(k0, k1, k2, k3, k4, tweak, counter, buffer)

    usedInts = 0
    counter += 1
  }

  override def setSeed(seed: Int): Unit = ???

  override def setSeed(seed: Long): Unit = ???

  override def setSeed(seed: Array[Int]): Unit = ???

  override def getSeed: Long = ???

  override def nextBytes(x: Array[Byte]): Unit = ???

  override def nextBoolean(): Boolean =
    (nextInt() ^ 1) == 0

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