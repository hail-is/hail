package is.hail

import is.hail.collection.ByteArrayBuilder
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.utils.implicits.toRichBoolean

import scala.collection.compat._
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.control.ControlThrowable

import java.io._
import java.lang.reflect.Method
import java.net.{URI, URLClassLoader}
import java.nio.charset.StandardCharsets
import java.security.SecureRandom
import java.util.Base64

import org.json4s.{Extraction, Formats, JObject, JValue}
import org.json4s.JsonAST.JString
import org.json4s.jackson.JsonMethods

package utils {
  trait Truncatable {
    def truncate: String
    def strings: (String, String)
  }

  sealed trait FlattenOrNull[C[_] >: Null] {
    def apply[T >: Null](b: mutable.Builder[T, C[T]], it: Iterable[Iterable[T]]): C[T] = {
      for (elt <- it) {
        if (elt == null)
          return null
        b ++= elt
      }
      b.result()
    }
  }

  sealed trait AnyFailAllFail[C[_]] {
    def apply[T](ts: IterableOnce[Option[T]])(implicit cbf: Factory[T, C[T]]): Option[C[T]] = {
      val b = cbf.newBuilder
      for (t <- ts.iterator)
        if (t.isEmpty)
          return None
        else
          b += t.get
      Some(b.result())
    }
  }

  sealed trait MapAccumulate[C[_], U] {
    def apply[T, S](
      a: Iterable[T],
      z: S,
    )(
      f: (T, S) => (U, S)
    )(implicit cbf: Factory[U, C[U]]
    ): C[U] = {
      val b = cbf.newBuilder
      var acc = z
      a.foreach { x =>
        val (y, newAcc) = f(x, acc)
        b += y
        acc = newAcc
      }
      b.result()
    }
  }

  class Lazy[A] private[utils] (f: => A) {
    private[this] var option: Option[A] = None

    def apply(): A =
      synchronized {
        option match {
          case Some(a) => a
          case None => val a = f; option = Some(a); a
        }
      }

    def force: A = apply()

    def isEvaluated: Boolean =
      synchronized {
        option.isDefined
      }
  }
}

package object utils extends ErrorHandling with Logging {

  type UtilsType = this.type

  def utilsPackageClass = getClass

  def format(s: String, substitutions: Any*): String =
    substitutions.zipWithIndex.foldLeft(s) { case (str, (value, i)) =>
      str.replace(s"@${i + 1}", value.toString)
    }

  def coerceToInt(l: Long): Int = {
    if (l > Int.MaxValue || l < Int.MinValue)
      fatal(s"int overflow: $l")
    l.toInt
  }

  def plural(n: Int, sing: String): String = plural(n.toLong, sing)

  def plural(n: Int, sing: String, plur: String): String = plural(n.toLong, sing, plur)

  def plural(n: Long, sing: String, plur: String = null): String =
    if (n == 1)
      sing
    else if (plur == null)
      sing + "s"
    else
      plur

  val noOp: () => Unit = () => ()

  def triangle(n: Int): Int = (n * (n + 1)) / 2

  def treeAggDepth(nPartitions: Int, branchingFactor: Int): Int = {
    require(nPartitions >= 0)
    require(branchingFactor > 0)

    if (nPartitions == 0)
      return 1

    math.ceil(math.log(nPartitions.toDouble) / math.log(branchingFactor.toDouble)).toInt
  }

  def simpleAssert(p: Boolean): Unit =
    if (!p) throw new AssertionError

  def optionCheckInRangeInclusive[A](low: A, high: A)(name: String, a: A)(implicit ord: Ordering[A])
    : Unit =
    if (ord.lt(a, low) || ord.gt(a, high)) {
      fatal(s"$name cannot lie outside [$low, $high]: $a")
    }

  def printTime[T](block: => T) = {
    val timed = time(block)
    println("time: " + formatTime(timed._2))
    timed._1
  }

  def time[A](f: => A): (A, Long) = {
    val t0 = System.nanoTime()
    val result = f
    val t1 = System.nanoTime()
    (result, t1 - t0)
  }

  final val msPerMinute = 60 * 1e3
  final val msPerHour = 60 * msPerMinute
  final val msPerDay = 24 * msPerHour

  def formatTime(dt: Long): String = {
    val tMilliseconds = dt / 1e6
    if (tMilliseconds < 1000)
      ("%.3f" + "ms").format(tMilliseconds)
    else if (tMilliseconds < msPerMinute)
      ("%.3f" + "s").format(tMilliseconds / 1e3)
    else if (tMilliseconds < msPerHour) {
      val tMins = (tMilliseconds / msPerMinute).toInt
      val tSec = (tMilliseconds % msPerMinute) / 1e3
      ("%d" + "m" + "%.1f" + "s").format(tMins, tSec)
    } else {
      val tHrs = (tMilliseconds / msPerHour).toInt
      val tMins = ((tMilliseconds % msPerHour) / msPerMinute).toInt
      val tSec = (tMilliseconds % msPerMinute) / 1e3
      ("%d" + "h" + "%d" + "m" + "%.1f" + "s").format(tHrs, tMins, tSec)
    }
  }

  def space[A](f: => A): (A, Long) = {
    val rt = Runtime.getRuntime
    System.gc()
    System.gc()
    val before = rt.totalMemory() - rt.freeMemory()
    val r = f
    System.gc()
    val after = rt.totalMemory() - rt.freeMemory()
    (r, after - before)
  }

  def printSpace[A](f: => A): A = {
    val (r, ds) = space(f)
    println("space: " + formatSpace(ds))
    r
  }

  def formatSpace(ds: Long, precision: Int = 2): String = {
    val absds = ds.abs
    val kib = 1024L
    val mib = kib * 1024
    val gib = mib * 1024
    val tib = gib * 1024

    val (div: Long, suffix: String) = if (absds < kib)
      (1L, "B")
    else if (absds < mib)
      (kib, "KiB")
    else if (absds < gib)
      (mib, "MiB")
    else if (absds < tib)
      (gib, "GiB")
    else
      (tib, "TiB")

    val num = formatDouble(absds.toDouble / div.toDouble, precision)
    s"$num $suffix"
  }

  def someIf[T](p: Boolean, x: => T): Option[T] =
    if (p) Some(x)
    else None

  def nullIfNot(p: Boolean, x: Any): Any =
    if (p) x
    else null

  def divOption(num: Double, denom: Double): Option[Double] =
    someIf(denom != 0, num / denom)

  def divNull(num: Double, denom: Double): java.lang.Double =
    if (denom == 0)
      null
    else
      num / denom

  val defaultTolerance = 1e-6

  def D_epsilon(a: Double, b: Double, tolerance: Double = defaultTolerance): Double =
    math.max(java.lang.Double.MIN_NORMAL, tolerance * math.max(math.abs(a), math.abs(b)))

  def D_==(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean =
    a == b || math.abs(a - b) <= D_epsilon(a, b, tolerance)

  def D_!=(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean =
    !(a == b) && math.abs(a - b) > D_epsilon(a, b, tolerance)

  def D_<(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean =
    !(a == b) && a - b < -D_epsilon(a, b, tolerance)

  def D_<=(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean =
    (a == b) || a - b <= D_epsilon(a, b, tolerance)

  def D_>(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean =
    !(a == b) && a - b > D_epsilon(a, b, tolerance)

  def D_>=(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean =
    (a == b) || a - b >= -D_epsilon(a, b, tolerance)

  def D0_==(x: Double, y: Double, tolerance: Double = defaultTolerance): Boolean =
    if (x.isNaN)
      y.isNaN
    else if (x.isPosInfinity)
      y.isPosInfinity
    else if (x.isNegInfinity)
      y.isNegInfinity
    else
      D_==(x, y, tolerance)

  def flushDouble(a: Double): Double =
    if (math.abs(a) < java.lang.Double.MIN_NORMAL) 0.0 else a

  def prettyIdentifier(str: String): String =
    if (str.matches("""[_a-zA-Z]\w*""")) str
    else s"`${StringEscapeUtils.escapeString(str, backticked = true)}`"

  def formatDouble(d: Double, precision: Int): String =
    s"%.${precision}f".format(d)

  def uriPath(uri: String): String =
    new URI(uri).getPath

  // NB: can't use Nothing here because it is not a super type of Null
  private object flattenOrNullInstance extends FlattenOrNull[Array]

  def flattenOrNull[C[_] >: Null] =
    flattenOrNullInstance.asInstanceOf[FlattenOrNull[C]]

  private object anyFailAllFailInstance extends AnyFailAllFail[Nothing]

  def anyFailAllFail[C[_]]: AnyFailAllFail[C] =
    anyFailAllFailInstance.asInstanceOf[AnyFailAllFail[C]]

  def uninitialized[T]: T = null.asInstanceOf[T]

  def unreachable[A]: A = throw new AssertionError("unreachable")

  private object mapAccumulateInstance extends MapAccumulate[Nothing, Nothing]

  def mapAccumulate[C[_], U] =
    mapAccumulateInstance.asInstanceOf[MapAccumulate[C, U]]

  def getIteratorSize[T](iterator: Iterator[T]): Long = {
    var count = 0L
    while (iterator.hasNext) {
      count += 1L
      iterator.next(): Unit
    }
    count
  }

  def getIteratorSizeWithMaxN[T](max: Long)(iterator: Iterator[T]): Long = {
    var count = 0L
    while (iterator.hasNext && count < max) {
      count += 1L
      iterator.next(): Unit
    }
    count
  }

  def lookupMethod(c: Class[_], method: String): Method = {
    try
      c.getDeclaredMethod(method)
    catch {
      case _: Exception =>
        assert(c != classOf[java.lang.Object])
        lookupMethod(c.getSuperclass, method)
    }
  }

  def invokeMethod(obj: AnyRef, method: String, args: AnyRef*): AnyRef = {
    val m = lookupMethod(obj.getClass, method)
    m.invoke(obj, args: _*)
  }

  def dictionaryOrdering[T](ords: Ordering[T]*): Ordering[T] = {
    new Ordering[T] {
      override def compare(x: T, y: T): Int = {
        var i = 0
        while (i < ords.size) {
          val v = ords(i).compare(x, y)
          if (v != 0)
            return v
          i += 1
        }
        return 0
      }
    }
  }

//  val DefaultFormats: Formats =
//    Serialization.formats(NoTypeHints) + GenericIndexedSeqSerializer

  def box(i: Int): java.lang.Integer = i

  def box(l: Long): java.lang.Long = l

  def box(f: Float): java.lang.Float = f

  def box(d: Double): java.lang.Double = d

  def box(b: Boolean): java.lang.Boolean = b

  def intArraySum(a: Array[Int]): Int = {
    var s = 0
    var i = 0
    while (i < a.length) {
      s += a(i)
      i += 1
    }
    s
  }

  def loadFromResource[T](file: String)(reader: (InputStream) => T): T = {
    val resourceStream = Thread.currentThread().getContextClassLoader.getResourceAsStream(file)
    assert(resourceStream != null, s"Error while locating file '$file'")

    try
      reader(resourceStream)
    finally
      resourceStream.close()
  }

  def roundWithConstantSum(a: Array[Double]): Array[Int] = {
    val withFloors = a.zipWithIndex.map { case (d, i) => (i, d, math.floor(d)) }
    val totalFractional = (withFloors.map { case (_, orig, floor) => orig - floor }.sum + 0.5).toInt
    withFloors
      .sortBy { case (_, orig, floor) => floor - orig }
      .zipWithIndex
      .map { case ((i, orig, _), iSort) =>
        if (iSort < totalFractional)
          (i, math.ceil(orig))
        else
          (i, math.floor(orig))
      }.sortBy(_._1).map(_._2.toInt)
  }

  def uniqueMinIndex(a: Array[Int]): java.lang.Integer = {
    def f(i: Int, m: Int, mi: Int, count: Int): java.lang.Integer = {
      if (i == a.length) {
        assert(count >= 1)
        if (count == 1)
          mi
        else
          null
      } else if (a(i) < m)
        f(i + 1, a(i), i, 1)
      else if (a(i) == m)
        f(i + 1, m, mi, count + 1)
      else
        f(i + 1, m, mi, count)
    }

    if (a.isEmpty)
      null
    else
      f(1, a(0), 0, 1)
  }

  def uniqueMaxIndex(a: Array[Int]): java.lang.Integer = {
    def f(i: Int, m: Int, mi: Int, count: Int): java.lang.Integer = {
      if (i == a.length) {
        assert(count >= 1)
        if (count == 1)
          mi
        else
          null
      } else if (a(i) > m)
        f(i + 1, a(i), i, 1)
      else if (a(i) == m)
        f(i + 1, m, mi, count + 1)
      else
        f(i + 1, m, mi, count)
    }

    if (a.isEmpty)
      null
    else
      f(1, a(0), 0, 1)
  }

  def digitsNeeded(i: Int): Int = {
    assert(i >= 0)
    i.toString.length
  }

  def lift[T, S](pf: PartialFunction[T, S]): T => Option[S] =
    pf.lift

  def using[R <: AutoCloseable, T](r: R)(consume: (R) => T): T = {
    var caught = false
    try
      consume(r)
    catch {
      case original: Exception =>
        caught = true
        try
          r.close()
        catch {
          case duringClose: Exception =>
            if (original == duringClose) {
              logger.info(
                s"""The exact same exception object, $original, was thrown by both
                   |the consumer and the close method. I will throw the original.""".stripMargin
              )
              throw original
            } else {
              duringClose.addSuppressed(original)
              throw duringClose
            }
        }
        throw original
    } finally
      if (!caught) {
        r.close()
      }
  }

  def singletonElement[T](it: Iterator[T]): T = {
    val x = it.next()
    assert(!it.hasNext)
    x
  }

  // return partition of the ith item
  def itemPartition(i: Int, n: Int, k: Int): Int = {
    assert(n >= 0)
    assert(k > 0)
    assert(i >= 0 && i < n)
    val minItemsPerPartition = n / k
    val r = n % k
    if (r == 0)
      i / minItemsPerPartition
    else {
      val maxItemsPerPartition = minItemsPerPartition + 1
      val crossover = maxItemsPerPartition * r
      if (i < crossover)
        i / maxItemsPerPartition
      else
        r + ((i - crossover) / minItemsPerPartition)
    }
  }

  def partition(n: Int, k: Int): Array[Int] = {
    if (k == 0) {
      assert(n == 0)
      return Array.empty[Int]
    }

    assert(n >= 0)
    assert(k > 0)
    val parts = Array.tabulate(k)(i => n / k + (i < (n % k)).toInt)
    assert(parts.sum == n)
    assert(parts.max - parts.min <= 1)
    parts
  }

  def partition(n: Long, k: Int): Array[Long] = {
    if (k == 0) {
      assert(n == 0)
      return Array.empty[Long]
    }

    assert(n >= 0)
    assert(k > 0)
    val parts = Array.tabulate(k)(i => n / k + (i < (n % k)).toLong)
    assert(parts.sum == n)
    assert(parts.max - parts.min <= 1)
    parts
  }

  def matchErrorToNone[T, U](f: (T) => U): (T) => Option[U] = (x: T) =>
    try
      Some(f(x))
    catch {
      case _: MatchError => None
    }

  def charRegex(c: Char): String = {
    // See: https://docs.oracle.com/javase/tutorial/essential/regex/literals.html
    val metacharacters = "<([{\\^-=$!|]})?*+.>"
    val s = c.toString
    if (metacharacters.contains(c))
      "\\" + s
    else
      s
  }

  def makeJavaMap[K, V](x: IterableOnce[(K, V)]): java.util.HashMap[K, V] = {
    val m = new java.util.HashMap[K, V]
    x.iterator.foreach { case (k, v) => m.put(k, v) }
    m
  }

  def makeJavaSet[K](x: IterableOnce[K]): java.util.HashSet[K] = {
    val m = new java.util.HashSet[K]
    x.iterator.foreach(m.add)
    m
  }

  def toMapFast[T, K, V](
    ts: IterableOnce[T]
  )(
    key: T => K,
    value: T => V,
  ): scala.collection.Map[K, V] = {
    val it = ts.iterator
    val m = mutable.Map[K, V]()
    while (it.hasNext) {
      val t = it.next()
      m.update(key(t), value(t))
    }
    m
  }

  def toMapIfUnique[K, K2, V](
    kvs: Iterable[(K, V)]
  )(
    keyBy: K => K2
  ): Either[Map[K2, Iterable[K]], Map[K2, V]] = {
    val grouped = kvs.groupBy(x => keyBy(x._1))

    val dupes = grouped.filter { case (_, m) => m.size != 1 }

    if (dupes.nonEmpty) {
      Left(dupes.map { case (k, m) => k -> m.map(_._1) })
    } else {
      Right(grouped
        .map { case (k, m) => k -> m.map(_._2).head }
        .toMap)
    }
  }

  def dumpClassLoader(cl: ClassLoader): Unit = {
    System.err.println(s"ClassLoader ${cl.getClass.getCanonicalName}:")
    cl match {
      case cl: URLClassLoader =>
        System.err.println(s"  ${cl.getURLs.mkString(" ")}")
      case _ =>
        System.err.println("  non-URLClassLoader")
    }
    val parent = cl.getParent
    if (parent != null)
      dumpClassLoader(parent)
  }

  def decompress(input: Array[Byte], size: Int): Array[Byte] =
    CompressionUtils.decompressZlib(input, size)

  def compress(bb: ByteArrayBuilder, input: Array[Byte]): Int =
    CompressionUtils.compressZlib(bb, input)

  def unwrappedApply[U, T](f: (U, T) => T): (U, Seq[T]) => T =
    if (f == null) null else { (s, ts) => f(s, ts(0)) }

  def unwrappedApply[U, T](f: (U, T, T) => T): (U, Seq[T]) => T = if (f == null) null
  else { (s, ts) =>
    val Seq(t1, t2) = ts
    f(s, t1, t2)
  }

  def unwrappedApply[U, T](f: (U, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null
  else { (s, ts) =>
    val Seq(t1, t2, t3) = ts
    f(s, t1, t2, t3)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null
  else { (s, ts) =>
    val Seq(t1, t2, t3, t4) = ts
    f(s, t1, t2, t3, t4)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null
  else { (s, ts) =>
    val Seq(t1, t2, t3, t4, t5) = ts
    f(s, t1, t2, t3, t4, t5)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null
  else { (s, ts) =>
    val Seq(arg1, arg2, arg3, arg4, arg5, arg6) = ts
    f(s, arg1, arg2, arg3, arg4, arg5, arg6)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null
  else { (s, ts) =>
    val Seq(arg1, arg2, arg3, arg4, arg5, arg6, arg7) = ts
    f(s, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
  }

  def isJavaIdentifier(id: String): Boolean = {
    if (!java.lang.Character.isJavaIdentifierStart(id.head))
      return false

    var i = 1
    while (i < id.length) {
      if (!java.lang.Character.isJavaIdentifierPart(id(i)))
        return false
      i += 1
    }

    true
  }

  def commonPrefix[T](left: IndexedSeq[T], right: IndexedSeq[T]): IndexedSeq[T] = {
    var i = 0
    while (i < left.length && i < right.length && left(i) == right(i))
      i += 1
    if (i == left.length)
      left
    else if (i == right.length)
      right
    else
      left.take(i)
  }

  def decomposeWithName(v: Any, name: String)(implicit formats: Formats): JObject = {
    val jo = Extraction.decompose(v).asInstanceOf[JObject]
    jo.merge(JObject("name" -> JString(name)))
  }

  def makeVirtualOffset(fileOffset: Long, blockOffset: Int): Long = {
    assert(fileOffset >= 0)
    assert(blockOffset >= 0)
    assert(blockOffset < 64 * 1024)
    (fileOffset << 16) | blockOffset
  }

  def virtualOffsetBlockOffset(offset: Long): Int =
    (offset & 0xffff).toInt

  def virtualOffsetCompressedOffset(offset: Long): Long =
    offset >> 16

  def tokenUrlSafe: String = {
    val bytes = new Array[Byte](32)
    val random = new SecureRandom()
    random.nextBytes(bytes)
    Base64.getUrlEncoder.encodeToString(bytes)
  }

  // mutates byteOffsets and returns the byte size
  def getByteSizeAndOffsets(
    byteSize: Array[Long],
    alignment: Array[Long],
    nMissingBytes: Long,
    byteOffsets: Array[Long],
  ): Long = {
    assert(byteSize.length == alignment.length)
    assert(byteOffsets.length == byteSize.length)
    val bp = new BytePacker()

    var offset: Long = nMissingBytes
    byteSize.indices.foreach { i =>
      val fSize = byteSize(i)
      val fAlignment = alignment(i)

      bp.getSpace(fSize, fAlignment) match {
        case Some(start) =>
          byteOffsets(i) = start
        case None =>
          val mod = offset % fAlignment
          if (mod != 0) {
            val shift = fAlignment - mod
            bp.insertSpace(shift, offset)
            offset += (fAlignment - mod)
          }
          byteOffsets(i) = offset
          offset += fSize
      }
    }
    offset
  }

  /** Merge the sorted `IndexedSeq`s `xs` and `ys` using comparison function `lt`. */
  def merge[A: ClassTag](xs: IndexedSeq[A], ys: IndexedSeq[A], lt: (A, A) => Boolean)
    : IndexedSeq[A] =
    (xs.length, ys.length) match {
      case (0, _) => ys
      case (_, 0) => xs
      case (n, m) =>
        val res = ArraySeq.newBuilder[A]
        res.sizeHint(n + m)

        var i = 0
        var j = 0
        while (i < n && j < m) {
          if (lt(xs(i), ys(j))) {
            res += xs(i)
            i += 1
          } else {
            res += ys(j)
            j += 1
          }
        }

        for (k <- i until n)
          res += xs(k)

        for (k <- j until m)
          res += ys(k)

        res.result()
    }

  def lazily[A](f: => A): Lazy[A] =
    new Lazy(f)

  implicit def evalLazy[A](f: Lazy[A]): A =
    f.force

  def jsonToBytes(v: JValue): Array[Byte] =
    JsonMethods.compact(v).getBytes(StandardCharsets.UTF_8)

  private[this] object Retry extends ControlThrowable

  def retry[A]: A = throw Retry

  def retryable[A](f: Int => A): A = {
    var attempts: Int = 0

    while (true)
      try return f(attempts)
      catch {
        case Retry =>
          attempts += 1
      }

    unreachable
  }
}
