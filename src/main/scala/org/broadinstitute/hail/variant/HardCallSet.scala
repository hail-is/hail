package org.broadinstitute.hail.variant

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._

import scala.collection.mutable.ArrayBuffer

object HardCallSet {
  def apply(vds: VariantDataset): HardCallSet = {

    def toDenseCallStream(gs: Iterable[Genotype]): DenseCallStream = {
      DenseCallStream( gs.iterator.map(g => g.call.map(_.gt).getOrElse(3)).toArray )
    }

    HardCallSet(
      vds.rdd.map{ case (v, va, gs) => (v, toDenseCallStream(gs)) },
      vds.nSamples)
  }

  def read(sqlContext: SQLContext, dirname: String): HardCallSet = {
    require(dirname.endsWith(".hcs"))
    import RichRow._

    // need a better suffix than .ser?
    val nSamples = readDataFile(dirname + "/metadata.ser", sqlContext.sparkContext.hadoopConfiguration) (_.readInt())

    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    new HardCallSet(df.rdd.map(r => (r.getVariant(0), DenseCallStream(r.getByteArray(1)))), nSamples)
  }
}

case class HardCallSet(rdd: RDD[(Variant, DenseCallStream)], nSamples: Int) {
  def write(sqlContext: SQLContext, dirname: String) {
    require(dirname.endsWith(".hcs"))
    import sqlContext.implicits._

    val hConf = rdd.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeDataFile(dirname + "/metadata.ser", hConf) (_.writeInt(nSamples))

    rdd.mapValues(_.a).toDF().write.parquet(dirname + "/rdd.parquet")
  }
}

object DenseCallStream {

  def apply(gts: Array[Int]): DenseCallStream = {

    val a = Array.ofDim[Byte]((gts.length + 3) / 4)

    var i = 0
    var j = 0
    while (i < gts.length - 4) {
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

    DenseCallStream(a)
  }
}


case class DenseCallStream(a: Array[Byte]) { //extends CallStream {

  def denseStats(y: DenseVector[Double] , n: Int): DenseStats = {

    var i = 0
    var j = 0

    val x = Array.ofDim[Double](n)
    var sumX = 0
    var sumXX = 0
    var sumXY = 0.0
    val missingRows = ArrayBuffer[Int]()

    val mask00000011 = 3
    val mask00001100 = 3 << 2
    val mask00110000 = 3 << 4
    val mask11000000 = 3 << 6

    def merge(i: Int, gt: Int) {
      gt match {
        case 0 =>
          x(i) = 0 // would it be faster to initialize to 0 and leave this case out?
        case 1 =>
          x(i) = 1
          sumX += 1
          sumXX += 1
          sumXY += y(i)
        case 2 =>
          x(i) = 2
          sumX += 2
          sumXX += 4
          sumXY += 2 * y(i)
        case 3 =>
          missingRows += i
      }
    }

    while (i < n - 4) {
      val b = a(j)
      merge(i,     b & mask00000011)
      merge(i + 1, b & mask00001100)
      merge(i + 2, b & mask00110000)
      merge(i + 3, b & mask11000000)

      i += 4
      j += 1
    }

    n - i match {
      case 1 =>  merge(i,     a(j) & mask00000011)
      case 2 =>  merge(i,     a(j) & mask00000011)
                 merge(i + 1, a(j) & mask00001100)
      case 3 =>  merge(i,     a(j) & mask00000011)
                 merge(i + 1, a(j) & mask00001100)
                 merge(i + 2, a(j) & mask00110000)
      case _ =>
    }

    val nMissing = missingRows.size
    val nPresent = n - nMissing
    val meanX = sumX.toDouble / nPresent

    for (r <- missingRows)
      x(r) = meanX

    val xx = sumXX + meanX * meanX * nMissing
    val xy = sumXY + meanX * missingRows.iterator.map(y(_)).sum

    DenseStats(DenseVector(x), sumXX, sumXY, nMissing)
  }

  def toBinaryString(b: Byte): String = {
    for (i <- 7 to 0 by -1) yield (b & (1 << i)) >> i
  }.mkString("")

  def toIntsString(b: Byte): String = {
    for (i <- 6 to 0 by -2) yield (b & (3 << i)) >> i
  }.mkString(":")

  def showBinary() = println(a.map(b => toBinaryString(b)).mkString("[", ", ", "]"))

  override def toString = a.map(b => toIntsString(b)).mkString("[", ", ", "]")
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