package is.hail

import is.hail.annotations.ExtendedOrdering
import is.hail.check.Gen
import is.hail.expr.ir.ByteArrayBuilder
import is.hail.io.fs.{FS, FileListEntry}
import org.apache.commons.io.output.TeeOutputStream
import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.fs.PathIOException
import org.apache.hadoop.mapred.FileSplit
import org.apache.hadoop.mapreduce.lib.input.{FileSplit => NewFileSplit}
import org.apache.log4j.Level
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, TaskContext}
import org.json4s.JsonAST.{JArray, JString}
import org.json4s.jackson.Serialization
import org.json4s.reflect.TypeInfo
import org.json4s.{Extraction, Formats, JObject, NoTypeHints, Serializer}

import java.io._
import java.lang.reflect.Method
import java.net.{URI, URLClassLoader}
import java.security.SecureRandom
import java.text.SimpleDateFormat
import java.util.concurrent.ExecutorService
import java.util.{Base64, Date}
import scala.collection.generic.CanBuildFrom
import scala.collection.mutable.ArrayBuffer
import scala.collection.{GenTraversableOnce, TraversableOnce, mutable}
import scala.language.{higherKinds, implicitConversions}
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}
import org.apache.spark.sql.Row

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
    def apply[T](ts: TraversableOnce[Option[T]])(implicit cbf: CanBuildFrom[Nothing, T, C[T]]): Option[C[T]] = {
      val b = cbf()
      for (t <- ts) {
        if (t.isEmpty)
          return None
        else
          b += t.get
      }
      Some(b.result())
    }
  }

  sealed trait MapAccumulate[C[_], U] {
    def apply[T, S](a: Iterable[T], z: S)(f: (T, S) => (U, S))
      (implicit uct: ClassTag[U], cbf: CanBuildFrom[Nothing, U, C[U]]): C[U] = {
      val b = cbf()
      var acc = z
      for ((x, i) <- a.zipWithIndex) {
        val (y, newAcc) = f(x, acc)
        b += y
        acc = newAcc
      }
      b.result()
    }
  }
}

package object utils extends Logging
  with richUtils.Implicits
  with NumericPairImplicits
  with utils.NumericImplicits
  with Py4jUtils
  with ErrorHandling {

  def utilsPackageClass = getClass

  def getStderrAndLogOutputStream[T](implicit tct: ClassTag[T]): OutputStream =
    new TeeOutputStream(new LoggerOutputStream(log, Level.ERROR), System.err)

  def format(s: String, substitutions: Any*): String = {
    substitutions.zipWithIndex.foldLeft(s) { case (str, (value, i)) =>
      str.replace(s"@${ i + 1 }", value.toString)
    }
  }

  def coerceToInt(l: Long): Int = {
    if (l > Int.MaxValue || l < Int.MinValue)
      fatal(s"int overflow: $l")
    l.toInt
  }

  def checkGzipOfGlobbedFiles(
    globPaths: Seq[String],
    fileListEntries: Array[FileListEntry],
    forceGZ: Boolean,
    gzAsBGZ: Boolean,
    maxSizeMB: Int = 128
  ) = {
    if (fileListEntries.isEmpty)
      fatal(s"arguments refer to no files: ${globPaths.toIndexedSeq}.")
    if (!gzAsBGZ) {
      fileListEntries.foreach { fileListEntry =>
        val path = fileListEntry.getPath
        if (path.endsWith(".gz"))
          checkGzippedFile(fileListEntry, forceGZ, false, maxSizeMB)
      }
    }
  }

  def checkGzippedFile(
    fileListEntry: FileListEntry,
    forceGZ: Boolean,
    gzAsBGZ: Boolean,
    maxSizeMB: Int = 128
  ) {
    if (!forceGZ && !gzAsBGZ)
      fatal(
        s"""Cannot load file '${fileListEntry.getPath}'
           |  .gz cannot be loaded in parallel. Is the file actually *block* gzipped?
           |  If the file is actually block gzipped (even though its extension is .gz),
           |  use the 'force_bgz' argument to treat all .gz file extensions as .bgz.
           |  If you are sure that you want to load a non-block-gzipped file serially
           |  on one core, use the 'force' argument.""".stripMargin)
    else if (!gzAsBGZ) {
      val fileSize = fileListEntry.getLen
      if (fileSize > 1024 * 1024 * maxSizeMB)
        warn(
          s"""file '${fileListEntry.getPath}' is ${ readableBytes(fileSize) }
             |  It will be loaded serially (on one core) due to usage of the 'force' argument.
             |  If it is actually block-gzipped, either rename to .bgz or use the 'force_bgz'
             |  argument.""".stripMargin)
    }
  }

  def plural(n: Long, sing: String, plur: String = null): String =
    if (n == 1)
      sing
    else if (plur == null)
      sing + "s"
    else
      plur

  val noOp: () => Unit = () => ()

  def square[T](d: T)(implicit ev: T => scala.math.Numeric[T]#Ops): T = d * d

  def triangle(n: Int): Int = (n * (n + 1)) / 2

  def treeAggDepth(nPartitions: Int, branchingFactor: Int): Int = {
    require(nPartitions >= 0)
    require(branchingFactor > 0)

    if (nPartitions == 0)
      return 1

    math.ceil(math.log(nPartitions) / math.log(branchingFactor)).toInt
  }

  def simpleAssert(p: Boolean) {
    if (!p) throw new AssertionError
  }

  def optionCheckInRangeInclusive[A](low: A, high: A)(name: String, a: A)(implicit ord: Ordering[A]): Unit =
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
    }
    else {
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
    if (p)
      Some(x)
    else
      None

  def nullIfNot(p: Boolean, x: Any): Any = {
    if (p)
      x
    else
      null
  }

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

  def D_==(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean = {
    a == b || math.abs(a - b) <= D_epsilon(a, b, tolerance)
  }

  def D_!=(a: Double, b: Double, tolerance: Double = defaultTolerance): Boolean = {
    !(a == b) && math.abs(a - b) > D_epsilon(a, b, tolerance)
  }

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

  def genBase: Gen[Char] = Gen.oneOf('A', 'C', 'T', 'G')

  def getPartNumber(fname: String): Int = {
    val partRegex = """.*/?part-(\d+).*""".r

    fname match {
      case partRegex(i) => i.toInt
      case _ => throw new PathIOException(s"invalid partition file '$fname'")
    }
  }

  def rowIterator(r: Row): Iterator[Any] = new Iterator[Any] {
    var idx: Int = 0
    def hasNext: Boolean = idx < r.size
    def next: Any = {
      val a = r(idx)
      idx += 1
      a
    }
  }

  // ignore size; atomic, like String
  def genDNAString: Gen[String] = Gen.stringOf(genBase)
    .resize(12)
    .filter(s => !s.isEmpty)

  def prettyIdentifier(str: String): String = {
    if (str.matches("""[_a-zA-Z]\w*"""))
      str
    else
      s"`${ StringEscapeUtils.escapeString(str, backticked = true) }`"
  }

  def formatDouble(d: Double, precision: Int): String = d.formatted(s"%.${ precision }f")

  def uriPath(uri: String): String = new URI(uri).getPath

  def removeFileProtocol(uriString: String): String = {
    val uri = new URI(uriString)
    if (uri.getScheme == "file") {
      uri.getPath
    } else {
      uri.toString
    }
  }

  // NB: can't use Nothing here because it is not a super type of Null
  private object flattenOrNullInstance extends FlattenOrNull[Array]

  def flattenOrNull[C[_] >: Null] =
    flattenOrNullInstance.asInstanceOf[FlattenOrNull[C]]

  private object anyFailAllFailInstance extends AnyFailAllFail[Nothing]

  def anyFailAllFail[C[_]]: AnyFailAllFail[C] =
    anyFailAllFailInstance.asInstanceOf[AnyFailAllFail[C]]

  def uninitialized[T]: T = null.asInstanceOf[T]

  private object mapAccumulateInstance extends MapAccumulate[Nothing, Nothing]

  def mapAccumulate[C[_], U] =
    mapAccumulateInstance.asInstanceOf[MapAccumulate[C, U]]

  /**
    * An abstraction for building an {@code Array} of known size. Guarantees a left-to-right traversal
    *
    * @param xs      the thing to iterate over
    * @param size    the size of array to allocate
    * @param key     given the source value and its source index, yield the target index
    * @param combine given the target value, the target index, the source value, and the source index, compute the new target value
    * @tparam A
    * @tparam B
    */
  def coalesce[A, B: ClassTag](xs: GenTraversableOnce[A])(size: Int, key: (A, Int) => Int, z: B)(combine: (B, A) => B): Array[B] = {
    val a = Array.fill(size)(z)

    for ((x, idx) <- xs.toIterator.zipWithIndex) {
      val k = key(x, idx)
      a(k) = combine(a(k), x)
    }

    a
  }

  def mapSameElements[K, V](l: Map[K, V], r: Map[K, V], valueEq: (V, V) => Boolean): Boolean = {
    def entryMismatchMessage(failures: TraversableOnce[(K, V, V)]): String = {
      require(failures.nonEmpty)
      val newline = System.lineSeparator()
      val sb = new StringBuilder
      sb ++= "The maps do not have the same entries:" + newline
      for (failure <- failures) {
        sb ++= s"  At key ${ failure._1 }, the left map has ${ failure._2 } and the right map has ${ failure._3 }" + newline
      }
      sb ++= s"  The left map is: $l" + newline
      sb ++= s"  The right map is: $r" + newline
      sb.result()
    }

    if (l.keySet != r.keySet) {
      println(
        s"""The maps do not have the same keys.
           |  These keys are unique to the left-hand map: ${ l.keySet -- r.keySet }
           |  These keys are unique to the right-hand map: ${ r.keySet -- l.keySet }
           |  The left map is: $l
           |  The right map is: $r
      """.stripMargin)
      false
    } else {
      val fs = Array.newBuilder[(K, V, V)]
      for ((k, lv) <- l) {
        val rv = r(k)
        if (!valueEq(lv, rv))
          fs += ((k, lv, rv))
      }
      val failures = fs.result()

      if (!failures.isEmpty) {
        println(entryMismatchMessage(failures))
        false
      } else {
        true
      }
    }
  }

  def getIteratorSize[T](iterator: Iterator[T]): Long = {
    var count = 0L
    while (iterator.hasNext) {
      count += 1L
      iterator.next()
    }
    count
  }

  def getIteratorSizeWithMaxN[T](max: Long)(iterator: Iterator[T]): Long = {
    var count = 0L
    while (iterator.hasNext && count < max) {
      count += 1L
      iterator.next()
    }
    count
  }

  def lookupMethod(c: Class[_], method: String): Method = {
    try {
      c.getDeclaredMethod(method)
    } catch {
      case _: Exception =>
        assert(c != classOf[java.lang.Object])
        lookupMethod(c.getSuperclass, method)
    }
  }

  def invokeMethod(obj: AnyRef, method: String, args: AnyRef*): AnyRef = {
    val m = lookupMethod(obj.getClass, method)
    m.invoke(obj, args: _*)
  }

  /*
   * Use reflection to get the path of a partition coming from a Parquet read.  This requires accessing Spark
   * internal interfaces.  It works with Spark 1 and 2 and doesn't depend on the location of the Parquet
   * package (parquet vs org.apache.parquet) which can vary between distributions.
   */
  def partitionPath(p: Partition): String = {
    p.getClass.getCanonicalName match {
      case "org.apache.spark.rdd.SqlNewHadoopPartition" =>
        val split = invokeMethod(invokeMethod(p, "serializableHadoopSplit"), "value").asInstanceOf[NewFileSplit]
        split.getPath.getName

      case "org.apache.spark.sql.execution.datasources.FilePartition" =>
        val files = invokeMethod(p, "files").asInstanceOf[Seq[_ <: AnyRef]]
        assert(files.length == 1)
        invokeMethod(files(0), "filePath").asInstanceOf[String]

      case "org.apache.spark.rdd.HadoopPartition" =>
        val split = invokeMethod(invokeMethod(p, "inputSplit"), "value").asInstanceOf[FileSplit]
        split.getPath.getName
    }
  }

  def dictionaryOrdering[T](ords: Ordering[T]*): Ordering[T] = {
    new Ordering[T] {
      def compare(x: T, y: T): Int = {
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

  val defaultJSONFormats: Formats = Serialization.formats(NoTypeHints) + GenericIndexedSeqSerializer

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
    val totalFractional = (withFloors.map { case (i, orig, floor) => orig - floor }.sum + 0.5).toInt
    withFloors
      .sortBy { case (_, orig, floor) => floor - orig }
      .zipWithIndex
      .map { case ((i, orig, floor), iSort) =>
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
    if (i < 10)
      1
    else
      1 + digitsNeeded(i / 10)
  }

  def partFile(numDigits: Int, i: Int): String = {
    val is = i.toString
    assert(is.length <= numDigits)
    "part-" + StringUtils.leftPad(is, numDigits, "0")
  }

  def partSuffix(ctx: TaskContext): String = {
    val rng = new java.security.SecureRandom()
    val fileUUID = new java.util.UUID(rng.nextLong(), rng.nextLong())
    s"${ ctx.stageId() }-${ ctx.partitionId() }-${ ctx.attemptNumber() }-$fileUUID"
  }

  def partFile(d: Int, i: Int, ctx: TaskContext): String = s"${ partFile(d, i) }-${ partSuffix(ctx) }"

  def mangle(strs: Array[String], formatter: Int => String = "_%d".format(_)): (Array[String], Array[(String, String)]) = {
    val b = new BoxedArrayBuilder[String]

    val uniques = new mutable.HashSet[String]()
    val mapping = new BoxedArrayBuilder[(String, String)]

    strs.foreach { s =>
      var smod = s
      var i = 0
      while (uniques.contains(smod)) {
        i += 1
        smod = s + formatter(i)
      }

      if (smod != s)
        mapping += s -> smod
      uniques += smod
      b += smod
    }

    b.result() -> mapping.result()
  }

  def lift[T, S](pf: PartialFunction[T, S]): (T) => Option[S] = pf.lift

  def flatLift[T, S](pf: PartialFunction[T, Option[S]]): (T) => Option[S] = pf.flatLift

  def optMatch[T, S](a: T)(pf: PartialFunction[T, S]): Option[S] = lift(pf)(a)

  def using[R <: AutoCloseable, T](r: R)(consume: (R) => T): T = {
    var caught = false
    try {
      consume(r)
    } catch {
      case original: Exception =>
        caught = true
        try {
          r.close()
        } catch {
          case duringClose: Exception =>
            if (original == duringClose) {
              log.info(s"""The exact same exception object, ${original}, was thrown by both
                          |the consumer and the close method. I will throw the original.""".stripMargin)
              throw original
            } else {
              duringClose.addSuppressed(original)
              throw duringClose
            }
        }
        throw original
    } finally {
      if (!caught) {
        r.close()
      }
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

  def matchErrorToNone[T, U](f: (T) => U): (T) => Option[U] = (x: T) => {
    try {
      Some(f(x))
    } catch {
      case _: MatchError => None
    }
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

  def ordMax[T](left: T, right: T, ord: ExtendedOrdering): T = {
    if (ord.gt(left, right))
      left
    else
      right
  }

  def ordMin[T](left: T, right: T, ord: ExtendedOrdering): T = {
    if (ord.lt(left, right))
      left
    else
      right
  }

  def makeJavaMap[K, V](x: TraversableOnce[(K, V)]): java.util.HashMap[K, V] = {
    val m = new java.util.HashMap[K, V]
    x.foreach { case (k, v) => m.put(k, v) }
    m
  }

  def makeJavaSet[K](x: TraversableOnce[K]): java.util.HashSet[K] = {
    val m = new java.util.HashSet[K]
    x.foreach(m.add)
    m
  }

  def toMapFast[T, K, V](
    ts: TraversableOnce[T]
  )(key: T => K,
    value: T => V
  ): collection.Map[K, V] = {
    val it = ts.toIterator
    val m = mutable.Map[K, V]()
    while (it.hasNext) {
      val t = it.next
      m.put(key(t), value(t))
    }
    m
  }

  def toMapIfUnique[K, K2, V](
    kvs: Traversable[(K, V)]
  )(keyBy: K => K2
  ): Either[Map[K2, Traversable[K]], Map[K2, V]] = {
    val grouped = kvs.groupBy(x => keyBy(x._1))

    val dupes = grouped.filter { case (k, m) => m.size != 1 }

    if (dupes.nonEmpty) {
      Left(dupes.map { case (k, m) => k -> m.map(_._1) })
    } else {
      Right(grouped
        .map { case (k, m) => k -> m.map(_._2).head }
        .toMap)
    }
  }

  def dumpClassLoader(cl: ClassLoader) {
    System.err.println(s"ClassLoader ${ cl.getClass.getCanonicalName }:")
    cl match {
      case cl: URLClassLoader =>
        System.err.println(s"  ${ cl.getURLs.mkString(" ") }")
      case _ =>
        System.err.println("  non-URLClassLoader")
    }
    val parent = cl.getParent
    if (parent != null)
      dumpClassLoader(parent)
  }

  def writeNativeFileReadMe(fs: FS, path: String): Unit = {
    val dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss")

    using(new OutputStreamWriter(fs.create(path + "/README.txt"))) { out =>
      out.write(
        s"""This folder comprises a Hail (www.hail.is) native Table or MatrixTable.
           |  Written with version ${ HailContext.get.version }
           |  Created at ${ dateFormat.format(new Date()) }""".stripMargin)
    }
  }

  def decompress(input: Array[Byte], size: Int): Array[Byte] = CompressionUtils.decompressZlib(input, size)

  def compress(bb: ByteArrayBuilder, input: Array[Byte]): Int = CompressionUtils.compressZlib(bb, input)

  def unwrappedApply[U, T](f: (U, T) => T): (U, Seq[T]) => T = if (f == null) null else { (s, ts) =>
    f(s, ts(0))
  }

  def unwrappedApply[U, T](f: (U, T, T) => T): (U, Seq[T]) => T = if (f == null) null else { (s, ts) =>
    val Seq(t1, t2) = ts
    f(s, t1, t2)
  }

  def unwrappedApply[U, T](f: (U, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null else { (s, ts) =>
    val Seq(t1, t2, t3) = ts
    f(s, t1, t2, t3)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null else { (s, ts) =>
    val Seq(t1, t2, t3, t4) = ts
    f(s, t1, t2, t3, t4)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null else { (s, ts) =>
    val Seq(t1, t2, t3, t4, t5) = ts
    f(s, t1, t2, t3, t4, t5)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null else { (s, ts) =>
    val Seq(arg1, arg2, arg3, arg4, arg5, arg6) = ts
    f(s, arg1, arg2, arg3, arg4, arg5, arg6)
  }

  def unwrappedApply[U, T](f: (U, T, T, T, T, T, T, T) => T): (U, Seq[T]) => T = if (f == null) null else { (s, ts) =>
    val Seq(arg1, arg2, arg3, arg4, arg5, arg6, arg7) = ts
    f(s, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
  }

  def drainInputStreamToOutputStream(
    is: InputStream,
    os: OutputStream
  ): Unit = {
    val buffer = new Array[Byte](1024)
    var length = is.read(buffer)
    while (length != -1) {
      os.write(buffer, 0, length);
      length = is.read(buffer)
    }
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

  def virtualOffsetBlockOffset(offset: Long): Int = {
    (offset & 0xFFFF).toInt
  }

  def virtualOffsetCompressedOffset(offset: Long): Long = {
    offset >> 16
  }

  def tokenUrlSafe(n: Int): String = {
    val bytes = new Array[Byte](32)
    val random = new SecureRandom()
    random.nextBytes(bytes)
    Base64.getUrlEncoder.encodeToString(bytes)
  }

  // mutates byteOffsets and returns the byte size
  def getByteSizeAndOffsets(byteSize: Array[Long], alignment: Array[Long], nMissingBytes: Long, byteOffsets: Array[Long]): Long = {
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

  /**
   * Merge the sorted `IndexedSeq`s `xs` and `ys` using comparison function `lt`.
   */
  def merge[A](xs: IndexedSeq[A], ys: IndexedSeq[A], lt: (A, A) => Boolean): IndexedSeq[A] =
    (xs.length, ys.length) match {
      case (0, _) => ys
      case (_, 0) => xs
      case (n, m) =>

        val res = new ArrayBuffer[A](n + m)

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

        for (k <- i until n) {
          res += xs(k)
        }

        for (k <- j until m) {
          res += ys(k)
        }

        res
    }


  /**
   * Run (task, key) pairs on the `executor`, returning some `F` of the
   * failures and an `IndexedSeq` of the successes with their corresponding
   * key.
   */
  def runAll[F[_], A](executor: ExecutorService)
                     (accum: (F[Throwable], (Throwable, Int)) => F[Throwable])
                     (init: F[Throwable])
                     (tasks: IndexedSeq[(() => A, Int)])
  : (F[Throwable], IndexedSeq[(A, Int)]) = {

    var err = init
    val buffer = new mutable.ArrayBuffer[(A, Int)](tasks.length)

    tasks
      .map { case (t, k) => (executor.submit(() => Try(t())), k) }
      .foreach { case (f, k) =>
        f.get() match {
          case Success(v) =>
            buffer += ((v, k))

          case Failure(t) =>
            err = accum(err, (t, k))
        }
      }

    (err, buffer)
  }

  def runAllKeepFirstError[A](executor: ExecutorService)
  : IndexedSeq[(() => A, Int)] => (Option[Throwable], IndexedSeq[(A, Int)]) =
    runAll[Option, A](executor) { case (opt, (e, _)) => opt.orElse(Some(e)) } (None)
}

// FIXME: probably resolved in 3.6 https://github.com/json4s/json4s/commit/fc96a92e1aa3e9e3f97e2e91f94907fdfff6010d
object GenericIndexedSeqSerializer extends Serializer[IndexedSeq[_]] {
  val IndexedSeqClass = classOf[IndexedSeq[_]]

  override def serialize(implicit format: Formats) = {
    case seq: IndexedSeq[_] => JArray(seq.map(Extraction.decompose).toList)
  }

  override def deserialize(implicit format: Formats) = {
    case (TypeInfo(IndexedSeqClass, parameterizedType), JArray(xs)) =>
      val typeInfo = TypeInfo(parameterizedType
        .map(_.getActualTypeArguments()(0))
        .getOrElse(throw new RuntimeException("No type parameter info for type IndexedSeq"))
        .asInstanceOf[Class[_]],
        None)
      xs.map(x => Extraction.extract(x, typeInfo)).toArray[Any]
  }
}
