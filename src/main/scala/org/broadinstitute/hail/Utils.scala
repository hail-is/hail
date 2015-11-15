package org.broadinstitute.hail

import java.io._
import java.net.URI
import breeze.linalg.operators.{OpSub, OpAdd}
import org.apache.hadoop
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.rdd.RDD
import org.scalacheck.Gen
import org.scalacheck.Arbitrary._
import scala.collection.mutable
import scala.language.implicitConversions
import breeze.linalg.{Vector => BVector, DenseVector => BDenseVector, SparseVector => BSparseVector}
import org.apache.spark.mllib.linalg.{Vector => SVector, DenseVector => SDenseVector, SparseVector => SSparseVector}
import scala.reflect.ClassTag
import org.broadinstitute.hail.Utils._
import scala.reflect.runtime.currentMirror
import scala.tools.reflect.ToolBox

// FIXME AnyVal in Scala 2.11
class RichVector[T](v: Vector[T]) {
  def zipExact[T2](v2: Iterable[T2]): Vector[(T, T2)] = {
    val i = v.iterator
    val i2 = v2.iterator
    new Iterator[(T, T2)] {
      def hasNext: Boolean = {
        assert(i.hasNext == i2.hasNext)
        i.hasNext
      }

      def next() = (i.next(), i2.next())
    }.toVector
  }

  def zipWith[T2, V](v2: Iterable[T2], f: (T, T2) => V): Vector[V] = {
    val i = v.iterator
    val i2 = v2.iterator
    new Iterator[V]() {
      def hasNext = i.hasNext && i2.hasNext

      def next() = f(i.next(), i2.next())
    }.toVector
  }

  def zipWithExact[T2, V](v2: Iterable[T2], f: (T, T2) => V): Vector[V] = {
    val i = v.iterator
    val i2 = v2.iterator
    new Iterator[V] {
      def hasNext: Boolean = {
        assert(i.hasNext == i2.hasNext)
        i.hasNext
      }

      def next() = f(i.next(), i2.next())
    }.toVector
  }

  def zipWithAndIndex[T2, V](v2: Iterable[T2], f: (T, T2, Int) => V): Vector[V] = {
    val i = v.iterator
    val i2 = v2.iterator
    val i3 = Iterator.from(0)
    new Iterator[V] {
      def hasNext = i.hasNext && i2.hasNext

      def next() = f(i.next(), i2.next(), i3.next())
    }.toVector
  }

  def zipWith[T2, T3, V](v2: Iterable[T2], v3: Iterable[T3], f: (T, T2, T3) => V): Vector[V] = {
    val i = v.iterator
    val i2 = v2.iterator
    val i3 = v3.iterator
    new Iterator[V] {
      def hasNext = i.hasNext && i2.hasNext && i3.hasNext

      def next() = f(i.next(), i2.next(), i3.next())
    }.toVector
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

  def foreach2[T2](v2: Iterable[T2], f: (T, T2) => Unit) {
    val i = a.iterator
    val i2 = v2.iterator
    while (i.hasNext && i2.hasNext)
      f(i.next(), i2.next())
  }

  // FIXME unify with Vector zipWith above
  def zipWith[T2, V](v2: Iterable[T2], f: (T, T2) => V)(implicit vct: ClassTag[V]): Array[V] = {
    val i = a.iterator
    val i2 = v2.iterator
    new Iterator[V] {
      def hasNext = i.hasNext && i2.hasNext

      def next() = f(i.next(), i2.next())
    }.toArray
  }

  def zipWith[T2, T3, V](v2: Iterable[T2], v3: Iterable[T3], f: (T, T2, T3) => V)(implicit vct: ClassTag[V]): Array[V] = {
    val i = a.iterator
    val i2 = v2.iterator
    val i3 = v3.iterator
    new Iterator[V] {
      def hasNext = i.hasNext && i2.hasNext && i3.hasNext

      def next() = f(i.next(), i2.next(), i3.next())
    }.toArray
  }
}

class RichRDD[T](val r: RDD[T]) extends AnyVal {
  def countByValueRDD()(implicit tct: ClassTag[T]): RDD[(T, Int)] = r.map((_, 1)).reduceByKey(_ + _)

  def writeTable(filename: String, header: String = null) {
    if (header != null)
      writeTextFile(filename + ".header", r.sparkContext.hadoopConfiguration) {_.write(header)}
    hadoopDelete(filename, r.sparkContext.hadoopConfiguration, true)
    r.saveAsTextFile(filename)
  }
}

class RichIndexedRow(val r: IndexedRow) extends AnyVal {

  def -(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector - that)

  def +(that: BVector[Double]): IndexedRow = new IndexedRow(r.index, r.vector + that)
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
  def tsvAppend[T](v: Option[T]) {
    v match {
      case Some(x) => sb.append(x)
      case None => sb.append("NA")
    }
  }
}

object Utils {

  implicit def toRichMap[K, V](m: Map[K, V]): RichMap[K, V] = new RichMap(m)

  implicit def toRichRDD[T](r: RDD[T])(implicit tct: ClassTag[T]): RichRDD[T] = new RichRDD(r)

  implicit def toRichVector[T](v: Vector[T]): RichVector[T] = new RichVector(v)

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

  def warning(msg: String) {
    System.err.println("hail: warning: " + msg)
  }

  def fatal(msg: String): Nothing = {
    System.err.println("hail: fatal: " + msg)
    sys.exit(1)
  }

  def fail() {
    assert(false)
  }

  def hadoopFS(filename: String, hConf: hadoop.conf.Configuration): hadoop.fs.FileSystem =
    hadoop.fs.FileSystem.get(new URI(filename), hConf)

  def hadoopCreate(filename: String, hConf: hadoop.conf.Configuration): hadoop.fs.FSDataOutputStream =
    hadoopFS(filename, hConf).create(new hadoop.fs.Path(filename))

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
    lines: Traversable[String], header: String = null) {
    writeTextFile(filename, hConf) {
      fw =>
        if (header != null) fw.write(header)
        lines.foreach(fw.write)
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

  def toTSVString(a: Any): String = a match {
    case o: Option[Any] => o.map(toTSVString).getOrElse("NA")
    case _ => a.toString
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

  def eval[T](t: String): T = {
    val toolbox = currentMirror.mkToolBox()
    val ast = toolbox.parse(t)
    toolbox.typeCheck(ast)
    toolbox.eval(ast).asInstanceOf[T]
  }

  def genOption[T](g: Gen[T], someFrequency: Int = 4): Gen[Option[T]] =
    Gen.frequency((1, Gen.const(None)),
      (someFrequency, g.map(Some(_))))

  def genNonnegInt: Gen[Int] = arbitrary[Int].map(_ & Int.MaxValue)

  def genBase: Gen[Char] = Gen.oneOf('A', 'C', 'T', 'G')

  def genDNAString: Gen[String] = Gen.buildableOf[String, Char](genBase)

}
