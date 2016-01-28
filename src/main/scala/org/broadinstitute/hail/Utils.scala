package org.broadinstitute.hail

import java.io._
import java.net.URI
import breeze.linalg.operators.{OpSub, OpAdd}
import org.apache.hadoop
import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.io.IOUtils._
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.driver.HailConfiguration
import org.scalacheck.Gen
import org.scalacheck.Arbitrary._
import scala.collection.{TraversableOnce, mutable}
import scala.language.implicitConversions
import breeze.linalg.{Vector => BVector, DenseVector => BDenseVector, SparseVector => BSparseVector}
import org.apache.spark.mllib.linalg.{Vector => SVector, DenseVector => SDenseVector, SparseVector => SSparseVector}
import scala.reflect.ClassTag
import org.broadinstitute.hail.Utils._

class RichIterable[T](val i: Iterable[T]) extends Serializable {
  def lazyMap[S](f: (T) => S): Iterable[S] = new Iterable[S] with Serializable {
    def iterator: Iterator[S] = new Iterator[S] {
      val it: Iterator[T] = i.iterator

      def hasNext: Boolean = it.hasNext

      def next(): S = f(it.next())
    }
  }

  def foreachBetween(f: (T) => Unit)(g: (Unit) => Unit) {
    var first = true
    i.foreach { elem =>
      if (first)
        first = false
      else
        g()
      f(elem)
    }
  }

  def lazyMapWith[T2, S](i2: Iterable[T2], f: (T, T2) => S): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator

        def hasNext: Boolean = it.hasNext && it2.hasNext

        def next(): S = f(it.next(), it2.next())
      }
    }

  def lazyFilterWith[T2](i2: Iterable[T2], p: (T, T2) => Boolean): Iterable[T] =
    new Iterable[T] with Serializable {
      def iterator: Iterator[T] = new Iterator[T] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator

        var pending: Boolean = false
        var pendingNext: T = _

        def hasNext: Boolean = {
          while (!pending && it.hasNext && it2.hasNext) {
            val n = it.next()
            val n2 = it2.next()
            if (p(n, n2)) {
              pending = true
              pendingNext = n
            }
          }
          pending
        }

        def next(): T = {
          assert(pending)
          pending = false
          pendingNext
        }
      }
    }

  def lazyFlatMap[S](f: (T) => TraversableOnce[S]): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        var current: Iterator[S] = Iterator.empty

        def hasNext: Boolean =
          if (current.hasNext)
            true
          else {
            if (it.hasNext) {
              current = f(it.next()).toIterator
              hasNext
            } else
              false
          }

        def next(): S = current.next()
      }
    }

  def lazyFlatMapWith[S, T2](i2: Iterable[T2], f: (T, T2) => TraversableOnce[S]): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator
        var current: Iterator[S] = Iterator.empty

        def hasNext: Boolean =
          if (current.hasNext)
            true
          else {
            if (it.hasNext && it2.hasNext) {
              current = f(it.next(), it2.next()).toIterator
              hasNext
            } else
              false
          }

        def next(): S = current.next()
      }
    }

  def areDistinct(): Boolean = {
    val seen = mutable.HashSet[T]()
    for (x <- i)
      if (seen(x))
        return false
      else
        seen += x
    true
  }
}

class RichHomogenousTuple1[T](val t: Tuple1[T]) extends AnyVal {
  def at(i: Int) = i match {
    case 1 => t._1
  }

  def insert(i: Int, x: T): (T, T) = i match {
    case 0 => (x, t._1)
    case 1 => (t._1, x)
  }

  def remove(i: Int): Unit = {
    require(i == 0)
  }
}

class RichHomogenousTuple2[T](val t: (T, T)) extends AnyVal {
  def at(i: Int): T = i match {
    case 1 => t._1
    case 2 => t._2
  }


  def insert(i: Int, x: T): (T, T, T) = i match {
    case 0 => (x, t._1, t._2)
    case 1 => (t._1, x, t._2)
    case 2 => (t._1, t._2, x)
  }

  def remove(i: Int): Tuple1[T] = i match {
    case 1 => Tuple1(t._2)
    case 2 => Tuple1(t._1)
  }
}

class RichHomogenousTuple3[T](val t: (T, T, T)) extends AnyVal {
  def at(i: Int): T = i match {
    case 1 => t._1
    case 2 => t._2
    case 3 => t._3
  }

  def insert(i: Int, x: T): (T, T, T, T) = i match {
    case 0 => (x, t._1, t._2, t._3)
    case 1 => (t._1, x, t._2, t._3)
    case 2 => (t._1, t._2, x, t._3)
    case 3 => (t._1, t._2, t._3, x)
  }

  def remove(i: Int): (T, T) = i match {
    case 1 => (t._2, t._3)
    case 2 => (t._1, t._3)
    case 3 => (t._1, t._2)
  }
}

class RichArrayBuilderOfByte(val b: mutable.ArrayBuilder[Byte]) extends AnyVal {
  def writeULEB128(x0: Int) {
    require(x0 >= 0)

    var x = x0
    var more = true
    while (more) {
      var c = x & 0x7F
      x = x >>> 7

      if (x == 0)
        more = false
      else
        c = c | 0x80

      assert(c >= 0 && c <= 255)
      b += c.toByte
    }
  }
}

class RichIteratorOfByte(val i: Iterator[Byte]) extends AnyVal {
  def readULEB128(): Int = {
    var x: Int = 0
    var shift: Int = 0
    var b: Byte = 0
    do {
      b = i.next()
      x = x | ((b & 0x7f) << shift)
      shift += 7
    } while ((b & 0x80) != 0)

    x
  }
}

// FIXME AnyVal in Scala 2.11
class RichArray[T](a: Array[T]) {
  def index: Map[T, Int] = a.zipWithIndex.toMap

  def areDistinct() = a.toIterable.areDistinct()
}

class RichRDD[T](val r: RDD[T]) extends AnyVal {
  def countByValueRDD()(implicit tct: ClassTag[T]): RDD[(T, Int)] = r.map((_, 1)).reduceByKey(_ + _)

  def writeTable(filename: String, header: Option[String] = None, deleteTmpFiles: Boolean = true) {
    val hConf = r.sparkContext.hadoopConfiguration
    val tmpFileName = hadoopGetTemporaryFile(HailConfiguration.tmpDir, hConf)
    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(filename)))
    val headerExt = codec.map(_.getDefaultExtension).getOrElse("")

    header.foreach { str =>
      writeTextFile(tmpFileName + ".header" + headerExt, r.sparkContext.hadoopConfiguration) { s =>
        s.write(str)
        s.write("\n")
      }
    }

    codec match {
      case Some(x) => r.saveAsTextFile(tmpFileName, x.getClass)
      case None => r.saveAsTextFile(tmpFileName)
    }

    val filesToMerge = header match {
      case Some(_) => Array(tmpFileName + ".header" + headerExt, tmpFileName)
      case None => Array(tmpFileName)
    }

    hadoopDelete(filename, hConf, recursive = true) // overwriting by default

    val (_, dt) = time {
      hadoopCopyMerge(filesToMerge, filename, hConf, deleteTmpFiles)
    }
    println("merge time: " + formatTime(dt))

    if (deleteTmpFiles) {
      hadoopDelete(tmpFileName + ".header" + headerExt, hConf, recursive = false)
      hadoopDelete(tmpFileName, hConf, recursive = true)
    }
  }
}

class RichIndexedRow(val r: IndexedRow) extends AnyVal {

  def -(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector - that)

  def +(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector + that)

  def :/(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector :/ that)
}

class RichEnumeration[T <: Enumeration](val e: T) extends AnyVal {
  def withNameOption(name: String): Option[T#Value] =
    e.values.find(_.toString == name)
}

class RichMap[K, V](val m: Map[K, V]) extends AnyVal {
  def mapValuesWithKeys[T](f: (K, V) => T): Map[K, T] = m map { case (k, v) => (k, f(k, v)) }

  def force = m.map(identity) // needed to make serializable: https://issues.scala-lang.org/browse/SI-7005
}

class RichOption[T](val o: Option[T]) extends AnyVal {
  def contains(v: T): Boolean = o.isDefined && o.get == v
}

class RichStringBuilder(val sb: mutable.StringBuilder) extends AnyVal {
  def tsvAppend(a: Any) {
    a match {
      case null | None => sb.append("NA")
      case Some(x) => tsvAppend(x)
      case d: Double => sb.append(d.formatted("%.4e"))
      case i: Iterable[_] =>
        var first = true
        i.foreach { x =>
          if (first)
            first = false
          else
            sb += ','
          tsvAppend(x)
        }
      case arr: Array[_] =>
        var first = true
        arr.foreach { x =>
          if (first)
            first = false
          else
            sb += ','
          tsvAppend(x)
        }
      case _ => sb.append(a)
    }
  }
}


class RichIterator[T](val it: Iterator[T]) extends AnyVal {
  def existsExactly1(p: (T) => Boolean): Boolean = {
    var n: Int = 0
    while (it.hasNext)
      if (p(it.next())) {
        n += 1
        if (n > 1)
          return false
      }
    n == 1
  }
}

object Utils {
  implicit def toRichMap[K, V](m: Map[K, V]): RichMap[K, V] = new RichMap(m)

  implicit def toRichRDD[T](r: RDD[T])(implicit tct: ClassTag[T]): RichRDD[T] = new RichRDD(r)

  implicit def toRichIterable[T](i: Iterable[T]): RichIterable[T] = new RichIterable(i)

  implicit def toRichTuple2[T](t: (T, T)): RichHomogenousTuple2[T] = new RichHomogenousTuple2(t)

  implicit def toRichTuple3[T](t: (T, T, T)): RichHomogenousTuple3[T] = new RichHomogenousTuple3(t)

  implicit def toRichArrayBuilderOfByte(t: mutable.ArrayBuilder[Byte]): RichArrayBuilderOfByte =
    new RichArrayBuilderOfByte(t)

  implicit def toRichIteratorOfByte(i: Iterator[Byte]): RichIteratorOfByte =
    new RichIteratorOfByte(i)

  implicit def richArray[T](a: Array[T]): RichArray[T] = new RichArray(a)

  implicit def toRichIndexedRow(r: IndexedRow): RichIndexedRow =
    new RichIndexedRow(r)

  implicit def toBDenseVector(v: SDenseVector): BDenseVector[Double] =
    new BDenseVector(v.values)

  implicit def toBSparseVector(v: SSparseVector): BSparseVector[Double] =
    new BSparseVector(v.indices, v.values, v.size)

  implicit def toBVector(v: SVector): BVector[Double] = v match {
    case v: SSparseVector => v
    case v: SDenseVector => v
  }

  implicit def toSDenseVector(v: BDenseVector[Double]): SDenseVector =
    new SDenseVector(v.toArray)

  implicit def toSSparseVector(v: BSparseVector[Double]): SSparseVector =
    new SSparseVector(v.length, v.array.index, v.array.data)

  implicit def toSVector(v: BVector[Double]): SVector = v match {
    case v: BDenseVector[Double] => v
    case v: BSparseVector[Double] => v
  }

  implicit object subBVectorSVector
    extends OpSub.Impl2[BVector[Double], SVector, BVector[Double]] {
    def apply(a: BVector[Double], b: SVector): BVector[Double] = a - toBVector(b)
  }

  implicit object subBVectorIndexedRow
    extends OpSub.Impl2[BVector[Double], IndexedRow, IndexedRow] {
    def apply(a: BVector[Double], b: IndexedRow): IndexedRow =
      new IndexedRow(b.index, a - toBVector(b.vector))
  }

  implicit object addBVectorSVector
    extends OpAdd.Impl2[BVector[Double], SVector, BVector[Double]] {
    def apply(a: BVector[Double], b: SVector): BVector[Double] = a + toBVector(b)
  }

  implicit object addBVectorIndexedRow
    extends OpAdd.Impl2[BVector[Double], IndexedRow, IndexedRow] {
    def apply(a: BVector[Double], b: IndexedRow): IndexedRow =
      new IndexedRow(b.index, a + toBVector(b.vector))
  }

  implicit def toRichEnumeration[T <: Enumeration](e: T): RichEnumeration[T] =
    new RichEnumeration(e)

  implicit def toRichOption[T](o: Option[T]): RichOption[T] =
    new RichOption[T](o)

  def plural(n: Int, sing: String, plur: String = null): String =
    if (n == 1)
      sing
    else if (plur == null)
      sing + "s"
    else
      plur

  def info(msg: String) {
    System.err.println("hail: info: " + msg)
  }

  def warning(msg: String) {
    System.err.println("hail: warning: " + msg)
  }

  def fatal(msg: String): Nothing = {
    System.err.println("hail: fatal: " + msg)
    sys.exit(1)
  }

  def fail(): Nothing = {
    assert(false)
    sys.exit(1)
  }

  def hadoopFS(filename: String, hConf: hadoop.conf.Configuration): hadoop.fs.FileSystem =
    hadoop.fs.FileSystem.get(new URI(filename), hConf)

  def hadoopCreate(filename: String, hConf: hadoop.conf.Configuration): OutputStream = {
    val fs = hadoopFS(filename, hConf)
    val hPath = new hadoop.fs.Path(filename)
    val os = fs.create(hPath)
    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = codecFactory.getCodec(hPath)

    if (codec != null)
      codec.createOutputStream(os)
    else
      os
  }

  def hadoopOpen(filename: String, hConf: hadoop.conf.Configuration): InputStream = {
    val fs = hadoopFS(filename, hConf)
    val hPath = new hadoop.fs.Path(filename)
    val is = fs.open(hPath)
    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = codecFactory.getCodec(hPath)
    if (codec != null)
      codec.createInputStream(is)
    else
      is
  }

  def hadoopMkdir(dirname: String, hConf: hadoop.conf.Configuration) {
    hadoopFS(dirname, hConf).mkdirs(new hadoop.fs.Path(dirname))
  }

  def hadoopDelete(filename: String, hConf: hadoop.conf.Configuration, recursive: Boolean) {
    hadoopFS(filename, hConf).delete(new hadoop.fs.Path(filename), recursive)
  }

  def hadoopGetTemporaryFile(tmpdir: String, hConf: hadoop.conf.Configuration, nChar: Int = 10,
    prefix: Option[String] = None, suffix: Option[String] = None): String = {

    val destFS = hadoopFS(tmpdir, hConf)
    val prefixString = if (prefix.isDefined) prefix + "-" else ""
    val suffixString = if (suffix.isDefined) "." + suffix else ""

    def getRandomName: String = {
      val randomName = tmpdir + "/" + prefixString + scala.util.Random.alphanumeric.take(nChar).mkString + suffixString
      val fileExists = destFS.exists(new hadoop.fs.Path(randomName))

      if (!fileExists)
        randomName
      else
        getRandomName
    }
    getRandomName
  }

  def hadoopCopyMerge(srcFilenames: Array[String], destFilename: String, hConf: hadoop.conf.Configuration, deleteSource: Boolean = true) {

    val destPath = new hadoop.fs.Path(destFilename)
    val destFS = hadoopFS(destFilename, hConf)

    def globAndSort(filename: String): Array[FileStatus] = {
      val fs = hadoopFS(filename, hConf)
      val isDir = fs.getFileStatus(new hadoop.fs.Path(filename)).isDirectory
      val path = if (isDir) new hadoop.fs.Path(filename + "/*") else new hadoop.fs.Path(filename)

      fs.globStatus(path).sortWith(_.compareTo(_) < 0)
    }

    val srcFileStatuses = srcFilenames.flatMap {
      case p => globAndSort(p)
    }
    require(srcFileStatuses.forall {
      case fileStatus => fileStatus.getPath != destPath && fileStatus.isFile
    })

    val outputStream = destFS.create(destPath)

    try {
      for (fileStatus <- srcFileStatuses) {
        val srcFS = hadoopFS(fileStatus.getPath.toString, hConf)
        val inputStream = srcFS.open(fileStatus.getPath)
        try {
          copyBytes(inputStream, outputStream, hConf, false)
        } finally {
          inputStream.close()
        }
      }
    } finally {
      outputStream.close()
    }

    if (deleteSource) {
      srcFileStatuses.foreach {
        case fileStatus => hadoopDelete(fileStatus.getPath.toString, hConf, true)
      }
    }
  }

  def writeObjectFile[T](filename: String,
    hConf: hadoop.conf.Configuration)(f: (ObjectOutputStream) => T): T = {
    val oos = new ObjectOutputStream(hadoopCreate(filename, hConf))
    try {
      f(oos)
    } finally {
      oos.close()
    }
  }

  def readObjectFile[T](filename: String,
    hConf: hadoop.conf.Configuration)(f: (ObjectInputStream) => T): T = {
    val ois = new ObjectInputStream(hadoopOpen(filename, hConf))
    try {
      f(ois)
    } finally {
      ois.close()
    }
  }

  def readDataFile[T](filename: String,
    hConf: hadoop.conf.Configuration)(f: (DataInputStream) => T): T = {
    val dis = new DataInputStream(hadoopOpen(filename, hConf))
    try {
      f(dis)
    } finally {
      dis.close()
    }
  }

  def writeTextFile[T](filename: String,
    hConf: hadoop.conf.Configuration)(writer: (OutputStreamWriter) => T): T = {
    val oos = hadoopCreate(filename, hConf)
    val fw = new OutputStreamWriter(oos)
    try {
      writer(fw)
    } finally {
      fw.close()
    }
  }

  def writeDataFile[T](filename: String,
    hConf: hadoop.conf.Configuration)(writer: (DataOutputStream) => T): T = {
    val oos = hadoopCreate(filename, hConf)
    val dos = new DataOutputStream(oos)
    try {
      writer(dos)
    } finally {
      dos.close()
    }
  }

  def readFile[T](filename: String,
    hConf: hadoop.conf.Configuration)(reader: (InputStream) => T): T = {
    val is = hadoopOpen(filename, hConf)
    try {
      reader(is)
    } finally {
      is.close()
    }
  }

  def writeTable(filename: String, hConf: hadoop.conf.Configuration,
    lines: Traversable[String], header: Option[String] = None) {
    writeTextFile(filename, hConf) {
      fw =>
        header.map { h =>
          fw.write(h)
          fw.write('\n')
        }
        lines.foreach { line =>
          fw.write(line)
          fw.write('\n')
        }
    }
  }

  def square[T](d: T)(implicit ev: T => scala.math.Numeric[T]#Ops): T = d * d

  def simpleAssert(p: Boolean) {
    if (!p) throw new AssertionError
  }

  // FIXME Would be nice to have a version that averages three runs, perhaps even discarding an initial run. In this case the code block had better be functional!
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

  def someIf[T](p: Boolean, x: => T): Option[T] =
    if (p)
      Some(x)
    else
      None

  def divOption[T](num: T, denom: T)(implicit ev: T => Double): Option[Double] =
    someIf(denom != 0, ev(num) / denom)

  implicit def toRichStringBuilder(sb: mutable.StringBuilder): RichStringBuilder =
    new RichStringBuilder(sb)

  def D_epsilon(a: Double, b: Double, tolerance: Double = 1.0E-6): Double =
    math.max(java.lang.Double.MIN_NORMAL, tolerance * math.max(math.abs(a), math.abs(b)))

  def D_==(a: Double, b: Double, tolerance: Double = 1.0E-6): Boolean =
    math.abs(a - b) <= D_epsilon(a, b, tolerance)

  def D_!=(a: Double, b: Double, tolerance: Double = 1.0E-6): Boolean =
    math.abs(a - b) > D_epsilon(a, b, tolerance)

  def D_<(a: Double, b: Double, tolerance: Double = 1.0E-6): Boolean =
    a - b < -D_epsilon(a, b, tolerance)

  def D_<=(a: Double, b: Double, tolerance: Double = 1.0E-6): Boolean =
    a - b <= D_epsilon(a, b, tolerance)

  def D_>(a: Double, b: Double, tolerance: Double = 1.0E-6): Boolean =
    a - b > D_epsilon(a, b, tolerance)

  def D_>=(a: Double, b: Double, tolerance: Double = 1.0E-6): Boolean =
    a - b >= -D_epsilon(a, b, tolerance)

  def flushDouble(a: Double): Double =
    if (math.abs(a) < java.lang.Double.MIN_NORMAL) 0.0 else a

  def genOption[T](g: Gen[T], someFrequency: Int = 4): Gen[Option[T]] =
    Gen.frequency((1, Gen.const(None)),
      (someFrequency, g.map(Some(_))))

  def genNonnegInt: Gen[Int] = arbitrary[Int].map(_ & Int.MaxValue)

  def genBase: Gen[Char] = Gen.oneOf('A', 'C', 'T', 'G')

  def genDNAString: Gen[String] = Gen.buildableOf[String, Char](genBase)

  implicit def richIterator[T](it: Iterator[T]): RichIterator[T] = new RichIterator[T](it)
}
