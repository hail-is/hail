package org.broadinstitute.hail.utils

import java.io.{PrintWriter, Serializable}

import breeze.linalg.{DenseMatrix, Vector => BVector}
import org.apache.hadoop
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.Partitioner._
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partitioner, SparkContext}
import org.broadinstitute.hail.driver.HailConfiguration
import org.broadinstitute.hail.io.hadoop.{ByteArrayOutputFormat, BytesOnlyWritable}
import org.broadinstitute.hail.variant.{GenotypeStream, Variant}

import scala.collection.JavaConverters._
import scala.collection.{TraversableOnce, mutable}
import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag

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
    (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null) = {
    r.cogroup(other, partitioner).flatMapValues { pair =>
      val w = pair._2.headOption
      pair._1.map((_, w))
    }
  }

  def joinDistinct[W](other: RDD[(K, W)])
    (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null): RDD[(K, (V, W))] = joinDistinct(other, defaultPartitioner(r, other))

  def joinDistinct[W](other: RDD[(K, W)], partitioner: Partitioner)
    (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null) = {
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

class RichRow(r: Row) {

  def update(i: Int, a: Any): Row = {
    val arr = r.toSeq.toArray
    arr(i) = a
    Row.fromSeq(arr)
  }

  def getOrIfNull[T](i: Int, t: T): T = {
    if (r.isNullAt(i))
      t
    else
      r.getAs[T](i)
  }

  def getOption(i: Int): Option[Any] = {
    Option(r.get(i))
  }

  def getAsOption[T](i: Int): Option[T] = {
    if (r.isNullAt(i))
      None
    else
      Some(r.getAs[T](i))
  }

  def delete(i: Int): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    (0 until i).foreach(ab += r.get(_))
    (i + 1 until r.size).foreach(ab += r.get(_))
    val result = ab.result()
    if (result.isEmpty)
      null
    else
      Row.fromSeq(result)
  }

  def append(a: Any): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    ab ++= r.toSeq
    ab += a
    Row.fromSeq(ab.result())
  }

  def insertBefore(i: Int, a: Any): Row = {
    val ab = mutable.ArrayBuilder.make[Any]
    (0 until i).foreach(ab += r.get(_))
    ab += a
    (i until r.size).foreach(ab += r.get(_))
    Row.fromSeq(ab.result())
  }

  def getVariant(i: Int) = Variant.fromRow(r.getAs[Row](i))

  def getGenotypeStream(v: Variant, i: Int) = GenotypeStream.fromRow(v, r.getAs[Row](i))
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
