package org.broadinstitute.hail.variant

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._

import scala.collection.mutable.ArrayBuffer

object HardCallSet {
  def apply(vds: VariantDataset): HardCallSet = {
    val n = vds.nLocalSamples

    new HardCallSet(
      vds.rdd.map { case (v, va, gs) => (v, DenseCallStream(gs, n)) },
      vds.localSamples,
      vds.metadata.sampleIds)
  }

  def read(sqlContext: SQLContext, dirname: String): HardCallSet = {
    require(dirname.endsWith(".hcs"))
    import RichRow._

    val (localSamples, sampleIds) = readDataFile(dirname + "/sampleInfo.ser",
      sqlContext.sparkContext.hadoopConfiguration) {
      ds =>
        ds.readObject[(Array[Int],IndexedSeq[String])]
    }

    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    new HardCallSet(
      df.rdd.map(r => (r.getVariant(0), r.getDenseCallStream(1))),
      localSamples,
      sampleIds)
  }
}

case class HardCallSet(rdd: RDD[(Variant, DenseCallStream)], localSamples: Array[Int], sampleIds: IndexedSeq[String]) {
  def write(sqlContext: SQLContext, dirname: String) {
    require(dirname.endsWith(".hcs"))
    import sqlContext.implicits._

    val hConf = rdd.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeDataFile(dirname + "/sampleInfo.ser", hConf) {
      ss =>
        ss.writeObject((localSamples, sampleIds))
    }

    rdd.toDF().write.parquet(dirname + "/rdd.parquet")
  }

  def sparkContext: SparkContext = rdd.sparkContext

  def copy(rdd: RDD[(Variant, DenseCallStream)],
           localSamples: Array[Int] = localSamples,
           sampleIds: IndexedSeq[String] = sampleIds): HardCallSet =
    new HardCallSet(rdd, localSamples, sampleIds)

  def cache(): HardCallSet = copy(rdd = rdd.cache())
}

object DenseCallStream {

  def apply(gs: Iterable[Genotype], n: Int) =
    DenseCallStreamFromGtStream(gs.map(_.gt.getOrElse(3)), n: Int)

  def DenseCallStreamFromGtStream(gts: Iterable[Int], n: Int): DenseCallStream = {
    var x = Array.ofDim[Int](n)
    var sumX = 0
    var sumXX = 0
    var nMissing = 0

    for ((gt, i) <- gts.view.zipWithIndex)
      gt match {
        case 0 =>
         // x(i) = 0
        case 1 =>
          x(i) = 1
          sumX += 1
          sumXX += 1
        case 2 =>
          x(i) = 2
          sumX += 2
          sumXX += 4
        case _ =>
          x(i) = 3
          nMissing += 1
      }

    val meanX = sumX.toDouble / (n - nMissing)

    // println(s"${x.mkString("[",",","]")}, sumX=$sumX, meanX=$meanX, sumXX=$sumXX")

    new DenseCallStream(
      denseByteArray(x),
      meanX,
      sumXX + meanX * meanX * nMissing,
      nMissing)
  }

  def denseByteArray(gts: Array[Int]): Array[Byte] = {

    val a = Array.ofDim[Byte]((gts.length + 3) / 4)

    var i = 0
    var j = 0
    while (i < gts.length - 3) {
      a(j) = (gts(i) | gts(i + 1) << 2 | gts(i + 2) << 4 | gts(i + 3) << 6).toByte
      i += 4
      j += 1
    }

    gts.length - i match {
      case 1 => a(j) = gts(i).toByte
      case 2 => a(j) = (gts(i) | gts(i + 1) << 2).toByte
      case 3 => a(j) = (gts(i) | gts(i + 1) << 2 | gts(i + 2) << 4).toByte
      case _ =>
    }

    a
  }
}


case class DenseCallStream(a: Array[Byte], meanX: Double, sumXX: Double, nMissing: Int) { //extends CallStream {

  def denseStats(y: DenseVector[Double] , n: Int): GtVectorAndStats = {

    val x = Array.ofDim[Double](n)
    var sumXY = 0.0

    val mask00000011 = 3
    val mask00001100 = 3 << 2
    val mask00110000 = 3 << 4
    val mask11000000 = 3 << 6

    def merge(i: Int, gt: Int) {
      gt match {
        case 0 =>
          // x(i) = 0.0
        case 1 =>
          x(i) = 1.0
          sumXY += y(i)
        case 2 =>
          x(i) = 2.0
          sumXY += 2 * y(i)
        case missing => // FIXME: Is this equivalent to _?
          x(i) = this.meanX
          sumXY += this.meanX * y(i)
      }
    }

    var i = 0
    var j = 0

    while (i < n - 3) {
      val b = a(j)
      merge(i,      b & mask00000011)
      merge(i + 1, (b & mask00001100) >> 2)
      merge(i + 2, (b & mask00110000) >> 4)
      merge(i + 3, (b & mask11000000) >> 6)

      i += 4
      j += 1
    }

    n - i match {
      case 1 =>  merge(i,      a(j) & mask00000011)
      case 2 =>  merge(i,      a(j) & mask00000011)
                 merge(i + 1, (a(j) & mask00001100) >> 2)
      case 3 =>  merge(i,      a(j) & mask00000011)
                 merge(i + 1, (a(j) & mask00001100) >> 2)
                 merge(i + 2, (a(j) & mask00110000) >> 4)
      case _ =>
    }

    GtVectorAndStats(DenseVector(x), sumXX, sumXY, nMissing)
  }

  def toBinaryString(b: Byte): String = {
    for (i <- 7 to 0 by -1) yield (b & (1 << i)) >> i
  }.mkString("")

  def toIntsString(b: Byte): String = {
    for (i <- 6 to 0 by -2) yield (b & (3 << i)) >> i
  }.mkString(":")

  def showBinary() = println(a.map(b => toBinaryString(b)).mkString("[", ", ", "]"))

  override def toString = s"${a.map(b => toIntsString(b)).mkString("[", ", ", "]")}, $meanX, $sumXX, $nMissing"
}

case class GtVectorAndStats(x: breeze.linalg.Vector[Double], xx: Double, xy: Double, nMissing: Int)

object SparseCallStream {

  def apply(gs: Iterable[Genotype], n: Int) =
    SparseCallStreamFromGtStream(gs.map(_.gt.getOrElse(3)), n: Int)

  def SparseCallStreamFromGtStream(gts: Iterable[Int], n: Int): SparseCallStream = {
    var rowX = ArrayBuffer[Int]()
    var valX = ArrayBuffer[Int]()
    var sumX = 0
    var sumXX = 0
    var nMissing = 0

    for ((gt, i) <- gts.view.zipWithIndex)
      gt match {
        case 0 =>
        // x(i) = 0
        case 1 =>
          rowX += i
          valX += 1
          sumX += 1
          sumXX += 1
        case 2 =>
          rowX += i
          valX += 2
          sumX += 2
          sumXX += 4
        case _ =>
          rowX += i
          valX += 3
          nMissing += 1
      }

    val meanX = sumX.toDouble / (n - nMissing)

    // println(s"${x.mkString("[",",","]")}, sumX=$sumX, meanX=$meanX, sumXX=$sumXX")

    new SparseCallStream(
      sparseByteArray(rowX.toArray, valX.toArray),
      meanX,
      sumXX + meanX * meanX * nMissing,
      nMissing)
  }

  def sparseByteArray(rows: Array[Int], gts: Array[Int]): Array[Byte] = {

    val a = Array.ofDim[Byte]((gts.length + 3) / 4)

    var i = 0
    var j = 0
    while (i < gts.length - 3) {
      a(j) = (gts(i) | gts(i + 1) << 2 | gts(i + 2) << 4 | gts(i + 3) << 6).toByte
      i += 4
      j += 1
    }

    gts.length - i match {
      case 1 => a(j) = gts(i).toByte
      case 2 => a(j) = (gts(i) | gts(i + 1) << 2).toByte
      case 3 => a(j) = (gts(i) | gts(i + 1) << 2 | gts(i + 2) << 4).toByte
      case _ =>
    }

    a
  }
}


case class SparseCallStream(a: Array[Byte], meanX: Double, sumXX: Double, nMissing: Int) { //extends CallStream {

  def sparseStats(y: DenseVector[Double] , n: Int): GtVectorAndStats = {

    //FIXME: these don't need to be buffers...can compute length from meanX, sumXX, etc
    val rowX = ArrayBuffer[Int]()
    val valX = ArrayBuffer[Double]()
    var sumXY = 0.0

    val mask00000011 = 3
    val mask00001100 = 3 << 2
    val mask00110000 = 3 << 4
    val mask11000000 = 3 << 6

    def merge(i: Int, gt: Int) {
      gt match {
        case 0 =>
        // x(i) = 0.0
        case 1 =>
          rowX += i
          valX += 1.0
          sumXY += y(i)
        case 2 =>
          rowX += i
          valX += 2.0
          sumXY += 2 * y(i)
        case 3 =>
          rowX += i
          valX += this.meanX
          sumXY += this.meanX * y(i)
      }
    }


    //FIXME: rewrite all the rest for sparse
    var i = 0
    var j = 0

    while (i < n - 3) {
      val b = a(j)
      merge(i,      b & mask00000011)
      merge(i + 1, (b & mask00001100) >> 2)
      merge(i + 2, (b & mask00110000) >> 4)
      merge(i + 3, (b & mask11000000) >> 6)

      i += 4
      j += 1
    }

    n - i match {
      case 1 =>  merge(i,      a(j) & mask00000011)
      case 2 =>  merge(i,      a(j) & mask00000011)
                 merge(i + 1, (a(j) & mask00001100) >> 2)
      case 3 =>  merge(i,      a(j) & mask00000011)
                 merge(i + 1, (a(j) & mask00001100) >> 2)
                 merge(i + 2, (a(j) & mask00110000) >> 4)
      case _ =>
    }

    GtVectorAndStats(new SparseVector(n, rowX.toArray, valX.toArray), sumXX, sumXY, nMissing)
  }

  def toBinaryString(b: Byte): String = {
    for (i <- 7 to 0 by -1) yield (b & (1 << i)) >> i
  }.mkString("")

  def toIntsString(b: Byte): String = {
    for (i <- 6 to 0 by -2) yield (b & (3 << i)) >> i
  }.mkString(":")

  def showBinary() = println(a.map(b => toBinaryString(b)).mkString("[", ", ", "]"))

  override def toString = s"${a.map(b => toIntsString(b)).mkString("[", ", ", "]")}, $meanX, $sumXX, $nMissing"
}

/*
abstract class CallStream {
//  def toLinRegBuilder: LinRegBuilder = {
//  }

  def toBinaryString(b: Byte): String = {
    for (i <- 7 to 0 by -1) yield (b & (1 << i)) >> i
  }.mkString("")

  def toIntsString(b: Byte): String = {
    for (i <- 6 to 0 by -2) yield (b & (3 << i)) >> i
  }.mkString(":")

  def encodeGtByte(gts: Array[Int], s: Int): Byte =
    (if (s + 3 < gts.length)
      gts(s) | gts(s + 1) << 2 | gts(s + 2) << 4 | gts(s + 3) << 6
    else if (s + 3 == gts.length)
      gts(s) | gts(s + 1) << 2 | gts(s + 2) << 4
    else if (s + 2 == gts.length)
      gts(s) | gts(s + 1) << 2
    else
      gts(s)
      ).toByte
}
*/

/*
object SparseCalls {
  def apply(gts: Array[Int]): SparseCalls = {
    SparseCalls(Array[Byte]())
  }

  def encodeBytes(sparseGts: Array[(Int, Int)]): Iterator[Byte] = {
    val gtByte = CallStream.encodeGtByte(sparseGts.map(_._2), 0)
    val lByte = encodeLByte(sparseGts.map(_._1))
    val sBytes = Iterator(0)
  }

  def encodeLByte(ss: Array[Int]): Byte = ss.map(nBytesForInt).map(CallStream.encodeGtByte)
}

case class SparseCalls(a: Array[Byte]) extends CallStream {
  def iterator = Iterator()
}
*/