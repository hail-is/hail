package org.broadinstitute.hail

import java.io._
import java.net.URI

import breeze.linalg.operators.{OpAdd, OpSub}
import breeze.linalg.{DenseMatrix, DenseVector => BDenseVector, SparseVector => BSparseVector, Vector => BVector}
import htsjdk.samtools.util.BlockCompressedStreamConstants
import org.apache.commons.lang.StringEscapeUtils
import org.apache.hadoop
import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.io.IOUtils._
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.Partitioner._
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.{DenseVector => SDenseVector, SparseVector => SSparseVector, Vector => SVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{AccumulableParam, SparkContext, Partitioner, SparkEnv, TaskContext}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.driver.HailConfiguration
import org.broadinstitute.hail.io.hadoop.{ByteArrayOutputFormat, BytesOnlyWritable}
import org.broadinstitute.hail.utils.RichRow
import org.broadinstitute.hail.variant.Variant
import org.seqdoop.hadoop_bam.util.BGZFCodec
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ListBuffer
import scala.collection.{TraversableOnce, mutable}
import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.util.Random
import scala.collection.JavaConverters._

final class ByteIterator(val a: Array[Byte]) {
  var i: Int = 0

  def hasNext: Boolean = i < a.length

  def next(): Byte = {
    val r = a(i)
    i += 1
    r
  }

  def readULEB128(): Int = {
    var b: Byte = next()
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = next()
      x |= ((b & 0x7f) << shift)
      shift += 7
    }

    x
  }

  def readSLEB128(): Int = {
    var b: Byte = next()
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = next()
      x |= ((b & 0x7f) << shift)
      shift += 7
    }

    // sign extend
    if (shift < 32
      && (b & 0x40) != 0)
      x = (x << (32 - shift)) >> (32 - shift)

    x
  }
}

class RichIterable[T](val i: Iterable[T]) extends Serializable {
  def lazyMap[S](f: (T) => S): Iterable[S] = new Iterable[S] with Serializable {
    def iterator: Iterator[S] = new Iterator[S] {
      val it: Iterator[T] = i.iterator

      def hasNext: Boolean = it.hasNext

      def next(): S = f(it.next())
    }
  }

  def foreachBetween(f: (T) => Unit)(g: () => Unit) {
    richIterator(i.iterator).foreachBetween(f)(g)
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

  def lazyMapWith2[T2, T3, S](i2: Iterable[T2], i3: Iterable[T3], f: (T, T2, T3) => S): Iterable[S] =
    new Iterable[S] with Serializable {
      def iterator: Iterator[S] = new Iterator[S] {
        val it: Iterator[T] = i.iterator
        val it2: Iterator[T2] = i2.iterator
        val it3: Iterator[T3] = i3.iterator

        def hasNext: Boolean = it.hasNext && it2.hasNext && it3.hasNext

        def next(): S = f(it.next(), it2.next(), it3.next())
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

  def duplicates(): Set[T] = {
    val dups = mutable.HashSet[T]()
    val seen = mutable.HashSet[T]()
    for (x <- i)
      if (seen(x))
        dups += x
      else
        seen += x
    dups.toSet
  }
}

class RichArrayBuilderOfByte(val b: mutable.ArrayBuilder[Byte]) extends AnyVal {
  def writeULEB128(x0: Int) {
    require(x0 >= 0, s"tried to write negative ULEB value `${x0}'")

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

  def writeSLEB128(x0: Int) {
    var more = true
    var x = x0
    while (more) {
      var c = x & 0x7f
      x >>= 7

      if ((x == 0
        && (c & 0x40) == 0)
        || (x == -1
        && (c & 0x40) == 0x40))
        more = false
      else
        c |= 0x80

      b += c.toByte
    }
  }
}

class RichIteratorOfByte(val i: Iterator[Byte]) extends AnyVal {
  /*
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

  def readSLEB128(): Int = {
    var shift: Int = 0
    var x: Int = 0
    var b: Byte = 0
    do {
      b = i.next()
      x |= ((b & 0x7f) << shift)
      shift += 7
    } while ((b & 0x80) != 0)

    // sign extend
    if (shift < 32
      && (b & 0x40) != 0)
      x = (x << (32 - shift)) >> (32 - shift)

    x
  }
  */


}

// FIXME AnyVal in Scala 2.11
class RichArray[T](a: Array[T]) {
  def index: Map[T, Int] = a.zipWithIndex.toMap

  def areDistinct() = a.toIterable.areDistinct()

  def duplicates(): Set[T] = a.toIterable.duplicates()
}

class RichOrderedArray[T: Ordering](a: Array[T]) {
  def isIncreasing: Boolean = a.toSeq.isIncreasing

  def isSorted: Boolean = a.toSeq.isSorted
}

class RichOrderedSeq[T: Ordering](s: Seq[T]) {

  import scala.math.Ordering.Implicits._

  def isIncreasing: Boolean = s.isEmpty || (s, s.tail).zipped.forall(_ < _)

  def isSorted: Boolean = s.isEmpty || (s, s.tail).zipped.forall(_ <= _)
}


class RichSparkContext(val sc: SparkContext) extends AnyVal {
  def textFilesLines(files: Array[String], f: String => Unit = s => (),
    nPartitions: Int = sc.defaultMinPartitions): RDD[Line] = {
    files.foreach(f)
    sc.union(
      files.map(file =>
        sc.textFileLines(file, nPartitions)))
  }

  def textFileLines(file: String, nPartitions: Int = sc.defaultMinPartitions): RDD[Line] =
    sc.textFile(file, nPartitions)
      .map(l => Line(l, None, file))
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
      case Some(_) => Array(tmpFileName + ".header" + headerExt, tmpFileName + "/part-*")
      case None => Array(tmpFileName + "/part-*")
    }

    hadoopDelete(filename, hConf, recursive = true) // overwriting by default

    val (_, dt) = time {
      hadoopCopyMerge(filesToMerge, filename, hConf, deleteTmpFiles)
    }
    info(s"while writing:\n    $filename\n  merge time: ${formatTime(dt)}")

    if (deleteTmpFiles) {
      hadoopDelete(tmpFileName + ".header" + headerExt, hConf, recursive = false)
      hadoopDelete(tmpFileName, hConf, recursive = true)
    }
  }
}

class SpanningIterator[K, V](val it: Iterator[(K, V)]) extends Iterator[(K, Iterable[V])] {
  val bit = it.buffered
  var n: Option[(K, Iterable[V])] = None

  override def hasNext: Boolean = {
    if (n.isDefined) return true
    n = computeNext
    n.isDefined
  }

  override def next(): ((K, Iterable[V])) = {
    val result = n.get
    n = None
    result
  }

  def computeNext: (Option[(K, Iterable[V])]) = {
    var k: Option[K] = None
    val span: ListBuffer[V] = ListBuffer()
    while (bit.hasNext) {
      if (k.isEmpty) {
        val (k_, v_) = bit.next
        k = Some(k_)
        span += v_
      } else if (bit.head._1 == k.get) {
        span += bit.next._2
      } else {
        return Some((k.get, span))
      }
    }
    k.map((_, span))
  }
}

class RichRDDByteArray(val r: RDD[Array[Byte]]) extends AnyVal {
  def saveFromByteArrays(filename: String, header: Option[Array[Byte]] = None, deleteTmpFiles: Boolean = true) {
    val nullWritableClassTag = implicitly[ClassTag[NullWritable]]
    val bytesClassTag = implicitly[ClassTag[BytesOnlyWritable]]
    val hConf = r.sparkContext.hadoopConfiguration

    val tmpFileName = hadoopGetTemporaryFile(HailConfiguration.tmpDir, hConf)

    header.foreach { str =>
      writeDataFile(tmpFileName + ".header", r.sparkContext.hadoopConfiguration) { s =>
        s.write(str)
      }
    }

    val filesToMerge = header match {
      case Some(_) => Array(tmpFileName + ".header", tmpFileName + "/part-*")
      case None => Array(tmpFileName + "/part-*")
    }

    val rMapped = r.mapPartitions { iter =>
      val bw = new BytesOnlyWritable()
      iter.map { bb =>
        bw.set(new BytesWritable(bb))
        (NullWritable.get(), bw)
      }
    }

    RDD.rddToPairRDDFunctions(rMapped)(nullWritableClassTag, bytesClassTag, null)
      .saveAsHadoopFile[ByteArrayOutputFormat](tmpFileName)

    hadoopDelete(filename, hConf, recursive = true) // overwriting by default

    val (_, dt) = time {
      hadoopCopyMerge(filesToMerge, filename, hConf, deleteTmpFiles)
    }
    println("merge time: " + formatTime(dt))

    if (deleteTmpFiles) {
      hadoopDelete(tmpFileName + ".header", hConf, recursive = false)
      hadoopDelete(tmpFileName, hConf, recursive = true)
    }
  }
}

class RichPairRDD[K, V](val r: RDD[(K, V)]) extends AnyVal {

  def spanByKey()(implicit kct: ClassTag[K], vct: ClassTag[V]): RDD[(K, Iterable[V])] =
    r.mapPartitions(p => new SpanningIterator(p))

  def leftOuterJoinDistinct[W](other: RDD[(K, W)])
                              (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null): RDD[(K, (V, Option[W]))] = leftOuterJoinDistinct(other, defaultPartitioner(r, other))

  def leftOuterJoinDistinct[W](other: RDD[(K, W)], partitioner: Partitioner)
                              (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null)= {
    r.cogroup(other, partitioner).flatMapValues { pair =>
      val w = pair._2.headOption
      pair._1.map((_, w))
    }
  }

  def joinDistinct[W](other: RDD[(K, W)])
                     (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null): RDD[(K, (V, W))] = joinDistinct(other, defaultPartitioner(r, other))

  def joinDistinct[W](other: RDD[(K, W)], partitioner: Partitioner)
                     (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null)= {
    r.cogroup(other, partitioner).flatMapValues { pair =>
      for (v <- pair._1.iterator; w <- pair._2.iterator.take(1)) yield (v, w)
    }
  }
}

class RichIndexedRow(val r: IndexedRow) extends AnyVal {

  def -(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector - that)

  def +(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector + that)

  def :*(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector :* that)

  def :/(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector :/ that)
}

class RichEnumeration[T <: Enumeration](val e: T) extends AnyVal {
  def withNameOption(name: String): Option[T#Value] =
    e.values.find(_.toString == name)
}

class RichMutableMap[K, V](val m: mutable.Map[K, V]) extends AnyVal {
  def updateValue(k: K, default: V, f: (V) => V) {
    m += ((k, f(m.getOrElse(k, default))))
  }
}

class RichMap[K, V](val m: Map[K, V]) extends AnyVal {
  def mapValuesWithKeys[T](f: (K, V) => T): Map[K, T] = m map { case (k, v) => (k, f(k, v)) }

  def force = m.map(identity) // needed to make serializable: https://issues.scala-lang.org/browse/SI-7005
}

class RichOption[T](val o: Option[T]) extends AnyVal {
  def contains(v: T): Boolean = o.isDefined && o.get == v

  override def toString: String = o.toString
}

class RichStringBuilder(val sb: mutable.StringBuilder) extends AnyVal {
  def tsvAppend(a: Any) {
    a match {
      case null | None => sb.append("NA")
      case Some(x) => tsvAppend(x)
      case d: Double => sb.append(d.formatted("%.4e"))
      case v: Variant =>
        sb.append(v.contig)
        sb += ':'
        sb.append(v.start)
        sb += ':'
        sb.append(v.ref)
        sb += ':'
        sb.append(v.altAlleles.map(_.alt).mkString(","))
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

class RichIntPairTraversableOnce[V](val t: TraversableOnce[(Int, V)]) extends AnyVal {
  def reduceByKeyToArray(n: Int, zero: => V)(f: (V, V) => V)(implicit vct: ClassTag[V]): Array[V] = {
    val a = Array.fill[V](n)(zero)
    t.foreach { case (k, v) =>
      a(k) = f(a(k), v)
    }
    a
  }
}

class RichPairTraversableOnce[K, V](val t: TraversableOnce[(K, V)]) extends AnyVal {
  def reduceByKey(f: (V, V) => V): scala.collection.Map[K, V] = {
    val m = mutable.Map.empty[K, V]
    t.foreach { case (k, v) =>
      m.get(k) match {
        case Some(v2) => m += k -> f(v, v2)
        case None => m += k -> v
      }
    }
    m
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

  def foreachBetween(f: (T) => Unit)(g: () => Unit) {
    var first = true
    it.foreach { elem =>
      if (first)
        first = false
      else
        g()
      f(elem)
    }
  }

  def pipe(pb: ProcessBuilder,
    printHeader: (String => Unit) => Unit,
    printElement: (String => Unit, T) => Unit,
    printFooter: (String => Unit) => Unit): Iterator[String] = {

    val command = pb.command().asScala.mkString(" ")

    val proc = pb.start()

    // Start a thread to print the process's stderr to ours
    new Thread("stderr reader for " + command) {
      override def run() {
        for (line <- Source.fromInputStream(proc.getErrorStream).getLines) {
          System.err.println(line)
        }
      }
    }.start()

    // Start a thread to feed the process input from our parent's iterator
    new Thread("stdin writer for " + command) {
      override def run() {
        val out = new PrintWriter(proc.getOutputStream)

        printHeader(out.println)
        it.foreach(x => printElement(out.println, x))
        printFooter(out.println)
        out.close()
      }
    }.start()

    // Return an iterator that read lines from the process's stdout
    Source.fromInputStream(proc.getInputStream).getLines()
  }
}

class RichBoolean(val b: Boolean) extends AnyVal {
  def ==>(that: => Boolean): Boolean = !b || that

  def iff(that: Boolean): Boolean = b == that

  def toInt: Double = if (b) 1 else 0

  def toDouble: Double = if (b) 1d else 0d
}

trait Logging {
  @transient var log_ : Logger = null

  def log: Logger = {
    if (log_ == null)
      log_ = LoggerFactory.getLogger("Hail")
    log_
  }
}

class FatalException(msg: String) extends RuntimeException(msg)

class RichAny(val a: Any) extends AnyVal {
  def castOption[T](implicit ct: ClassTag[T]): Option[T] =
    if (ct.runtimeClass.isInstance(a))
      Some(a.asInstanceOf[T])
    else
      None
}

object RichDenseMatrixDouble {
  def horzcat(oms: Option[DenseMatrix[Double]]*): Option[DenseMatrix[Double]] = {
    val ms = oms.flatMap(m => m)
    if (ms.isEmpty)
      None
    else
      Some(DenseMatrix.horzcat(ms: _*))
  }
}


// Not supporting generic T because its difficult to do with ArrayBuilder and not needed yet. See:
// http://stackoverflow.com/questions/16306408/boilerplate-free-scala-arraybuilder-specialization
class RichDenseMatrixDouble(val m: DenseMatrix[Double]) extends AnyVal {
  def filterRows(keepRow: Int => Boolean): Option[DenseMatrix[Double]] = {
    val ab = new mutable.ArrayBuilder.ofDouble

    var nRows = 0
    for (row <- 0 until m.rows)
      if (keepRow(row)) {
        nRows += 1
        for (col <- 0 until m.cols)
          ab += m(row, col)
      }

    if (nRows > 0)
      Some(new DenseMatrix[Double](rows = nRows, cols = m.cols, data = ab.result(),
        offset = 0, majorStride = m.cols, isTranspose = true))
    else
      None
  }

  def filterCols(keepCol: Int => Boolean): Option[DenseMatrix[Double]] = {
    val ab = new mutable.ArrayBuilder.ofDouble

    var nCols = 0
    for (col <- 0 until m.cols)
      if (keepCol(col)) {
        nCols += 1
        for (row <- 0 until m.rows)
          ab += m(row, col)
      }

    if (nCols > 0)
      Some(new DenseMatrix[Double](rows = m.rows, cols = nCols, data = ab.result()))
    else
      None
  }
}

object TempDir {
  def apply(tmpdir: String, hConf: hadoop.conf.Configuration): TempDir = {
    while (true) {
      try {
        val dirname = tmpdir + "/hail." + Random.alphanumeric.take(12).mkString

        hadoopMkdir(dirname, hConf)

        val fs = hadoopFS(tmpdir, hConf)
        fs.deleteOnExit(new hadoop.fs.Path(dirname))

        return new TempDir(dirname)
      } catch {
        case e: IOException =>
        // try again
      }
    }

    // can't happen
    null
  }
}

class TempDir(val dirname: String) {
  var counter: Int = 0

  def relFile(relPath: String) = dirname + "/" + relPath

  def relPath(relPath: String) =
    new URI(relFile(relPath)).getPath

  def createTempFile(prefix: String = "", extension: String = ""): String = {
    val i = counter
    counter += 1

    val sb = new StringBuilder
    sb.append(prefix)
    if (prefix != "")
      sb += '.'
    sb.append("%05d".format(i))
    sb.append(extension)

    relFile(sb.result())
  }
}

object Utils extends Logging {
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
    val isBGZF = codec.exists(_.isInstanceOf[BGZFCodec])

    val srcFileStatuses = srcFilenames.flatMap(f => hadoopGlobAndSort(f, hConf))
    require(srcFileStatuses.forall {
      fileStatus => fileStatus.getPath != destPath && fileStatus.isFile
    })

    val outputStream = destFS.create(destPath)

    try {
      srcFileStatuses.foreach { fileStatus =>
        val srcFS = hadoopFS(fileStatus.getPath.toString, hConf)
        val inputStream = srcFS.open(fileStatus.getPath)
        try {
          copyBytes(inputStream, outputStream,
            fileStatus.getLen,
            false)
        } finally {
          inputStream.close()
        }
      }
      if (isBGZF) {
        outputStream.write(BlockCompressedStreamConstants.EMPTY_GZIP_BLOCK)
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

  case class Line(value: String, position: Option[Int], filename: String) {
    def transform[T](f: Line => T): T = {
      try {
        f(this)
      } catch {
        case e: Exception =>
          val lineToPrint =
            if (value.length > 62)
              value.take(59) + "..."
            else
              value
          val msg = if (e.isInstanceOf[FatalException])
            e.getMessage
          else
            s"caught $e"
          log.error(
            s"""
               |$filename${position.map(ln => ":" + (ln + 1)).getOrElse("")}: $msg
               |  offending line: $value""".stripMargin)
          fatal(
            s"""
               |$filename${position.map(ln => ":" + (ln + 1)).getOrElse("")}: $msg
               |  offending line: $lineToPrint""".stripMargin)
      }
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

  def simpleAssert(p: Boolean) {
    if (!p) throw new AssertionError
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

  def genBase: Gen[Char] = Gen.oneOf('A', 'C', 'T', 'G')

  // ignore size; atomic, like String
  def genDNAString: Gen[String] = Gen.buildableOf[String, Char](genBase)
    .resize(12)
    .filter(s => !s.isEmpty)

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

  def prettyIdentifier(str: String): String = {
    if (str.matches( """\p{javaJavaIdentifierStart}\p{javaJavaIdentifierPart}*"""))
      str
    else
      s"`${escapeString(str)}`"
  }

  def escapeString(str: String): String = StringEscapeUtils.escapeJava(str)

  def unescapeString(str: String): String = StringEscapeUtils.unescapeJava(str)

  def uriPath(uri: String): String = new URI(uri).getPath
}
