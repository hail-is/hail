package org.broadinstitute.hail

import java.io._
import java.net.URI

import breeze.linalg.operators.{OpAdd, OpSub}
import breeze.linalg.{DenseMatrix, DenseVector => BDenseVector, SparseVector => BSparseVector, Vector => BVector}
import org.apache.commons.lang.StringEscapeUtils
import org.apache.hadoop
import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.io.IOUtils._
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.{DenseVector => SDenseVector, SparseVector => SSparseVector, Vector => SVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{AccumulableParam, SparkContext}
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.io.compress.BGzipCodec

import scala.collection.{TraversableOnce, mutable}
import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag

package object utils extends Logging {

  class FatalException(msg: String) extends RuntimeException(msg)

  def plural(n: Int, sing: String, plur: String = null): String =
    if (n == 1)
      sing
    else if (plur == null)
      sing + "s"
    else
      plur

  def info(msg: String) {
    log.info(msg)
    System.err.println("hail: info: " + msg)
  }

  def warn(msg: String) {
    log.warn(msg)
    System.err.println("hail: warning: " + msg)
  }

  def error(msg: String) {
    log.error(msg)
    System.err.println("hail: error: " + msg)
  }

  def fatal(msg: String): Nothing = {
    throw new FatalException(msg)
  }

  def hadoopFS(filename: String, hConf: hadoop.conf.Configuration): hadoop.fs.FileSystem =
    new hadoop.fs.Path(filename).getFileSystem(hConf)

  private def hadoopCreate(filename: String, hConf: hadoop.conf.Configuration): OutputStream = {
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

  private def hadoopOpen(filename: String, hConf: hadoop.conf.Configuration): InputStream = {
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

  def hadoopExists(hConf: hadoop.conf.Configuration, files: String*): Boolean = {
    files.forall(filename => hadoopFS(filename, hConf).exists(new hadoop.fs.Path(filename)))
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

  def hadoopGlobAll(filenames: Iterable[String], hConf: hadoop.conf.Configuration): Array[String] = {
    filenames.iterator
      .flatMap { arg =>
        val fss = hadoopGlobAndSort(arg, hConf)
        val files = fss.map(_.getPath.toString)
        if (files.isEmpty)
          warn(s"`$arg' refers to no files")
        files
      }.toArray
  }

  def hadoopGlobAndSort(filename: String, hConf: hadoop.conf.Configuration): Array[FileStatus] = {
    val fs = hadoopFS(filename, hConf)
    val path = new hadoop.fs.Path(filename)

    val files = fs.globStatus(path)
    if (files == null)
      return Array.empty[FileStatus]

    files.sortWith(_.compareTo(_) < 0)
  }

  def hadoopCopyMerge(srcFilenames: Array[String], destFilename: String, hConf: hadoop.conf.Configuration, deleteSource: Boolean = true) {

    val destPath = new hadoop.fs.Path(destFilename)
    val destFS = hadoopFS(destFilename, hConf)

    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(destFilename)))
    val isBGzip = codec.exists(_.isInstanceOf[BGzipCodec])

    val srcFileStatuses = srcFilenames.flatMap(f => hadoopGlobAndSort(f, hConf))
    require(srcFileStatuses.forall {
      fileStatus => fileStatus.getPath != destPath && fileStatus.isFile
    })

    val outputStream = destFS.create(destPath)

    try {
      var i = 0
      while (i < srcFileStatuses.length) {
        val fileStatus = srcFileStatuses(i)
        val lenAdjust: Long = if (isBGzip && i < srcFileStatuses.length - 1)
          -28
        else
          0
        val srcFS = hadoopFS(fileStatus.getPath.toString, hConf)
        val inputStream = srcFS.open(fileStatus.getPath)
        try {
          copyBytes(inputStream, outputStream,
            fileStatus.getLen + lenAdjust,
            false)
        } finally {
          inputStream.close()
        }
        i += 1
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

  def hadoopStripCodec(s: String, conf: hadoop.conf.Configuration): String = {
    val path = new org.apache.hadoop.fs.Path(s)

    Option(new CompressionCodecFactory(conf)
      .getCodec(path))
      .map { case codec =>
        val ext = codec.getDefaultExtension
        assert(s.endsWith(ext))
        s.dropRight(ext.length)
      }.getOrElse(s)
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

  def hadoopFileStatus(filename: String, hConf: hadoop.conf.Configuration): FileStatus =
    hadoopFS(filename, hConf).getFileStatus(new hadoop.fs.Path(filename))

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


  def truncate(str: String, length: Int = 60): String = {
    if (str.length > 57)
      str.take(57) + " ..."
    else
      str
  }

  def readLines[T](filename: String, hConf: hadoop.conf.Configuration)(reader: (Iterator[Line] => T)): T = {
    readFile[T](filename, hConf) {
      is =>
        val lines = Source.fromInputStream(is)
          .getLines()
          .zipWithIndex
          .map {
            case (value, position) => Line(value, Some(position), filename)
          }
        reader(lines)
    }
  }

  def writeTable(filename: String, hConf: hadoop.conf.Configuration,
    lines: Traversable[String], header: Option[String] = None) {
    writeTextFile(filename, hConf) {
      fw =>
        header.foreach { h =>
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

  def triangle(n: Int): Int = (n * (n + 1)) / 2

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

  def formatSpace(ds: Long) = {
    val absds = ds.abs
    if (absds < 1e3)
      s"${ds}B"
    else if (absds < 1e6)
      s"${ds.toDouble / 1e3}KB"
    else if (absds < 1e9)
      s"${ds.toDouble / 1e6}MB"
    else if (absds < 1e12)
      s"${ds.toDouble / 1e9}GB"
    else
      s"${ds.toDouble / 1e12}TB"
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

  def divOption[T](num: T, denom: T)(implicit ev: T => Double): Option[Double] =
    someIf(denom != 0, ev(num) / denom)

  def divNull[T](num: T, denom: T)(implicit ev: T => Double): Any =
    nullIfNot(denom != 0, ev(num) / denom)

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

  def genBase: Gen[Char] = Gen.oneOf('A', 'C', 'T', 'G')

  // ignore size; atomic, like String
  def genDNAString: Gen[String] = Gen.buildableOf[String, Char](genBase)
    .resize(12)
    .filter(s => !s.isEmpty)

  def prettyIdentifier(str: String): String = {
    if (str.matches( """\p{javaJavaIdentifierStart}\p{javaJavaIdentifierPart}*"""))
      str
    else
      s"`${escapeString(str)}`"
  }

  def escapeString(str: String): String = StringEscapeUtils.escapeJava(str)

  def unescapeString(str: String): String = StringEscapeUtils.unescapeJava(str)

  def uriPath(uri: String): String = new URI(uri).getPath

  /*
     ##################################
     ##  Implicit conversions below  ##
     ##################################
  */

  implicit def toRichMap[K, V](m: Map[K, V]): RichMap[K, V] = new RichMap(m)

  implicit def toRichMutableMap[K, V](m: mutable.Map[K, V]): RichMutableMap[K, V] = new RichMutableMap(m)

  implicit def toRichSC(sc: SparkContext): RichSparkContext = new RichSparkContext(sc)

  implicit def toRichRDD[T](r: RDD[T])(implicit tct: ClassTag[T]): RichRDD[T] = new RichRDD(r)

  implicit def toRichPairRDD[K, V](r: RDD[(K, V)])(implicit kct: ClassTag[K],
    vct: ClassTag[V]): RichPairRDD[K, V] = new RichPairRDD(r)

  implicit def toRichRDDByteArray(r: RDD[Array[Byte]]): RichRDDByteArray = new RichRDDByteArray(r)

  implicit def toRichIterable[T](i: Iterable[T]): RichIterable[T] = new RichIterable(i)

  implicit def toRichArrayBuilderOfByte(t: mutable.ArrayBuilder[Byte]): RichArrayBuilderOfByte =
    new RichArrayBuilderOfByte(t)

  implicit def toRichIteratorOfByte(i: Iterator[Byte]): RichIteratorOfByte =
    new RichIteratorOfByte(i)

  implicit def toRichArray[T](a: Array[T]): RichArray[T] = new RichArray(a)

  implicit def toRichOrderedArray[T: Ordering](a: Array[T]): RichOrderedArray[T] = new RichOrderedArray(a)

  implicit def toRichOrderedSeq[T: Ordering](s: Seq[T]): RichOrderedSeq[T] = new RichOrderedSeq[T](s)

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


  implicit def toRichPairTraversableOnce[K, V](t: TraversableOnce[(K, V)]): RichPairTraversableOnce[K, V] =
    new RichPairTraversableOnce[K, V](t)

  implicit def toRichIntPairTraversableOnce[V](t: TraversableOnce[(Int, V)]): RichIntPairTraversableOnce[V] =
    new RichIntPairTraversableOnce[V](t)

  implicit def toRichDenseMatrixDouble(m: DenseMatrix[Double]): RichDenseMatrixDouble =
    new RichDenseMatrixDouble(m)

  implicit def richIterator[T](it: Iterator[T]): RichIterator[T] = new RichIterator[T](it)

  implicit def richBoolean(b: Boolean): RichBoolean = new RichBoolean(b)

  implicit def accumulableMapInt[K]: AccumulableParam[mutable.Map[K, Int], K] = new AccumulableParam[mutable.Map[K, Int], K] {
    def addAccumulator(r: mutable.Map[K, Int], t: K): mutable.Map[K, Int] = {
      r.updateValue(t, 0, _ + 1)
      r
    }

    def addInPlace(r1: mutable.Map[K, Int], r2: mutable.Map[K, Int]): mutable.Map[K, Int] = {
      for ((k, v) <- r2)
        r1.updateValue(k, 0, _ + v)
      r1
    }

    def zero(initialValue: mutable.Map[K, Int]): mutable.Map[K, Int] =
      mutable.Map.empty[K, Int]
  }

  implicit def toRichAny(a: Any): RichAny = new RichAny(a)

  implicit def toRichRow(r: Row): RichRow = new RichRow(r)

  implicit def toRichStringBuilder(sb: mutable.StringBuilder): RichStringBuilder =
    new RichStringBuilder(sb)
}
