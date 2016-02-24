package org.broadinstitute.hail.variant

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._

import scala.collection.mutable.ArrayBuffer

object HardCallSet {
  def apply(vds: VariantDataset, sparseCutoff: Double = .05): HardCallSet = {
    val n = vds.nLocalSamples

    new HardCallSet(
      vds.rdd.map { case (v, va, gs) => (v, CallStream(gs, n, sparseCutoff)) },
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
      df.rdd.map(r => (r.getVariant(0), r.getCallStream(1))),
      localSamples,
      sampleIds)
  }
}

case class HardCallSet(rdd: RDD[(Variant, CallStream)], localSamples: Array[Int], sampleIds: IndexedSeq[String]) {
  def write(sqlContext: SQLContext, dirname: String) {
    if (!dirname.endsWith(".hcs"))
      fatal("Hard call set directory must end with .hcs")
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

  def copy(rdd: RDD[(Variant, CallStream)],
           localSamples: Array[Int] = localSamples,
           sampleIds: IndexedSeq[String] = sampleIds): HardCallSet =
    new HardCallSet(rdd, localSamples, sampleIds)

  def cache(): HardCallSet = copy(rdd = rdd.cache())
}

case class GtVectorAndStats(x: breeze.linalg.Vector[Double], xx: Double, xy: Double, nMissing: Int)

object CallStream {
  def apply(gs: Iterable[Genotype], n: Int, sparseCutoff: Double): CallStream = {
    val sparsity = 1 - gs.count(_.isHomRef).toDouble / n
    if (sparsity < sparseCutoff)
      SparseCallStream(gs, n)
    else
      DenseCallStream(gs, n)
  }

  def toBinaryString(b: Byte): String = {for (i <- 7 to 0 by -1) yield (b & (1 << i)) >> i}.mkString("")

  def toIntsString(b: Byte): String = {for (i <- 6 to 0 by -2) yield (b & (3 << i)) >> i}.mkString(":")
}

abstract class CallStream {
  val a: Array[Byte]
  val meanX: Double
  val sumXX: Double
  val nMissing: Int
  val isSparse: Boolean

  def hardStats(y: DenseVector[Double] , n: Int): GtVectorAndStats

  import CallStream._

  def showBinary() = println(a.map(b => toBinaryString(b)).mkString("[", ", ", "]"))

  override def toString = s"${a.map(b => toBinaryString(b)).mkString("[", ", ", "]")}, $meanX, $sumXX, $nMissing"
}

object DenseCallStream {

  // FIXME: combine two defs in PR?
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

    val meanX = sumX.toDouble / (n - nMissing) // FIXME: deal with case of all missing

    // println(s"${x.mkString("[",",","]")}, sumX=$sumX, meanX=$meanX, sumXX=$sumXX")

    new DenseCallStream(
      denseByteArray(x),
      meanX,
      sumXX + meanX * meanX * nMissing,
      nMissing,
      false)
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


case class DenseCallStream(a: Array[Byte], meanX: Double, sumXX: Double, nMissing: Int, isSparse: Boolean = false) extends CallStream {

  def hardStats(y: DenseVector[Double] , n: Int): GtVectorAndStats = {

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
}

object SparseCallStream {

  def apply(gs: Iterable[Genotype], n: Int) =
    SparseCallStreamFromGtStream(gs.map(_.gt.getOrElse(3)), n: Int)

  // FIXME: switch to ArrayBuilder?
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

//    println(s"${rowX.toArray.mkString("[",",","]")}, ${valX.toArray.mkString("[",",","]")}, sumX=$sumX, meanX=$meanX, sumXX=${sumXX + meanX * meanX * nMissing}")

    new SparseCallStream(
      sparseByteArray(rowX.toArray, valX.toArray),
      meanX,
      sumXX + meanX * meanX * nMissing,
      nMissing)
  }

  // is it faster to do direct bit comparisons
  // FIXME: somewhere we should require that i is positive...
  def nBytesMinus1(i: Int): Int =
    if (i < 0x100)
      0
    else if (i < 0x10000)
      1
    else if (i < 0x1000000)
      2
    else
      3

  def sparseByteArray(rows: Array[Int], gts: Array[Int]): Array[Byte] = {

    val a = ArrayBuffer[Byte]()

    var i = 0 // row index
    var j = 1 // counter for lenByte index
    var l = 1 // current val lenByte index

    while (i < gts.length - 3) {
      a += (gts(i + 3) << 6 | gts(i + 2) << 4 | gts(i + 1) << 2 | gts(i)).toByte // gtByte
      a += 0 // lenByte

      for (k <- 0 until 4) {
        val r = rows(i + k)
        val m = nBytesMinus1(r)
        m match {
          case 0 => a += r.toByte
          case 1 => a += (r >> 8).toByte
                    a += r.toByte
          case 2 => a += (r >> 16).toByte
                    a += (r >> 8).toByte
                    a += r.toByte
          case _ => a += (r >> 24).toByte
                    a += (r >> 16).toByte
                    a += (r >> 8).toByte
                    a += r.toByte
        }
        a(l) = (a(l) | (m << (2 * k))).toByte  // faster to use (k << 1) ?
        j += m
      }
      j += 6 // 1[lenByte] + 4 * 1[nBytesMinus1] + 1[gtByte]
      l = j

      i += 4
    }

    val nGtLeft = gts.length - i

    nGtLeft match {
      case 1 => a += gts(i).toByte
                a += 0
      case 2 => a += (gts(i + 1) << 2 | gts(i)).toByte
                a += 0
      case 3 => a += (gts(i + 2) << 4 | gts(i + 1) << 2 | gts(i)).toByte
                a += 0
      case _ =>
    }

    for (k <- 0 until nGtLeft) {
      val r = rows(i + k)
      val m = nBytesMinus1(r)
      m match {
        case 0 => a += r.toByte
        case 1 => a += (r >> 8).toByte
                  a += r.toByte
        case 2 => a += (r >> 16).toByte
                  a += (r >> 8).toByte
                  a += r.toByte
        case _ => a += (r >> 24).toByte
                  a += (r >> 16).toByte
                  a += (r >> 8).toByte
                  a += r.toByte
      }
      a(l) = (a(l) | (m << (2 * k))).toByte  // faster to use (k << 1) ?
    }

    a.toArray
  }
}

case class SparseCallStream(a: Array[Byte], meanX: Double, sumXX: Double, nMissing: Int, isSparse: Boolean = true) extends CallStream {

  def hardStats(y: DenseVector[Double] , n: Int): GtVectorAndStats = {

    //FIXME: these don't need to be buffers...can compute length from meanX, sumXX, etc
    val rowX = ArrayBuffer[Int]()
    val valX = ArrayBuffer[Double]()
    var sumXY = 0.0

    val mask00000011 = 3
    val mask00001100 = 3 << 2
    val mask00110000 = 3 << 4
    val mask11000000 = 3 << 6

    def merge(r: Int, gt: Int) {
      gt match {
        case 0 =>
        // x(i) = 0.0
        case 1 =>
          rowX += r
          valX += 1.0
          sumXY += y(r)
        case 2 =>
          rowX += r
          valX += 2.0
          sumXY += 2 * y(r)
        case 3 =>
          rowX += r
          valX += this.meanX
          sumXY += this.meanX * y(r)
      }
    }

    // FIXME: Store with & 0xFF already applied
    def rowInt(k: Int, l: Int) =
      l match {
        case 0 => a(k) & 0xFF
        case 1 => (a(k) & 0xFF) << 8 | (a(k + 1) & 0xFF)
        case 2 => (a(k) & 0xFF) << 16 | (a(k + 1) & 0xFF) << 8 | (a(k + 2) & 0xFF)
        case _ => a(k) << 24 | (a(k + 1) & 0xFF) << 16 | (a(k + 2) & 0xFF) << 8 | (a(k + 3) & 0xFF)
    }

    var i = 0
    var j = 0

    while (j < a.length) {

      val gtByte = a(j)
      val gt1 =  gtByte & mask00000011
      val gt2 = (gtByte & mask00001100) >> 2
      val gt3 = (gtByte & mask00110000) >> 4
      val gt4 = (gtByte & mask11000000) >> 6

      val lenByte = a(j + 1)
      val l1 =  lenByte & mask00000011
      val l2 = (lenByte & mask00001100) >> 2
      val l3 = (lenByte & mask00110000) >> 4
      val l4 = (lenByte & mask11000000) >> 6

      j += 2

      if (gt4 != 0) {
        merge(rowInt(j, l1), gt1)
        j += l1 + 1
        merge(rowInt(j, l2), gt2)
        j += l2 + 1
        merge(rowInt(j, l3), gt3)
        j += l3 + 1
        merge(rowInt(j, l4), gt4)
        j += l4 + 1

        i += 4
      }
      else if (gt3 != 0) {
        merge(rowInt(j, l1), gt1)
        j += l1 + 1
        merge(rowInt(j, l2), gt2)
        j += l2 + 1
        merge(rowInt(j, l3), gt3)
        j += l3 + 1
      }
      else if (gt2 != 0) {
        merge(rowInt(j, l1), gt1)
        j += l1 + 1
        merge(rowInt(j, l2), gt2)
        j += l2 + 1
      }
      else {
        merge(rowInt(j, l1), gt1)
        j += l1 + 1
      }
    }

    GtVectorAndStats(new SparseVector(n, rowX.toArray, valX.toArray), sumXX, sumXY, nMissing)
  }
}
