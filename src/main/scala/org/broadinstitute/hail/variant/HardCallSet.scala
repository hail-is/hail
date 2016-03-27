package org.broadinstitute.hail.variant

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.RichRow._
import org.broadinstitute.hail.methods.CovariateData

import scala.collection.mutable

object HardCallSet {
  def apply(sqlContext: SQLContext, vds: VariantDataset, sparseCutoff: Double = .15, blockWidth: Int = 1000000): HardCallSet = {
    import sqlContext.implicits._

    val n = vds.nLocalSamples

    new HardCallSet(
      vds.rdd.map { case (v, va, gs) =>
        (v.start, v.ref, v.alt, CallStream(gs, n, sparseCutoff), "chr" + v.contig, v.start / blockWidth)
      }.toDF("start", "ref", "alt", "callStream", "contig", "block"),
      vds.localSamples,
      vds.metadata.sampleIds,
      sparseCutoff,
      blockWidth)
  }

  def read(sqlContext: SQLContext, dirname: String): HardCallSet = {
    require(dirname.endsWith(".hcs"))

    val (localSamples, sampleIds, sparseCutoff, blockWidth) = readDataFile(dirname + "/sampleInfo.ser",
      sqlContext.sparkContext.hadoopConfiguration) {
      ds =>
        ds.readObject[(Array[Int], IndexedSeq[String], Double, Int)]
    }

    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    df.printSchema()

    new HardCallSet(df,
      localSamples,
      sampleIds,
      sparseCutoff,
      blockWidth)
  }
}

case class HardCallSet(df: DataFrame,
  localSamples: Array[Int],
  sampleIds: IndexedSeq[String],
  sparseCutoff: Double,
  blockWidth: Int) {

  def rdd: RDD[(Variant, CallStream)] = {
    import RichRow._
    df.rdd.map(r =>
      (Variant(r.getString(4).drop(3), // chr
        r.getInt(0),
        r.getString(1),
        r.getString(2)),
        r.getCallStream(3)))
  }

  def write(sqlContext: SQLContext, dirname: String) {
    if (!dirname.endsWith(".hcs"))
      fatal("Hard call set directory must end with .hcs")

    val hConf = rdd.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeDataFile(dirname + "/sampleInfo.ser", hConf) {
      ss =>
        ss.writeObject((localSamples, sampleIds, sparseCutoff, blockWidth))
    }

    df.write
      .partitionBy("contig", "block")
      .parquet(dirname + "/rdd.parquet")
  }

  def sparkContext: SparkContext = rdd.sparkContext

  def copy(df: DataFrame,
    localSamples: Array[Int] = localSamples,
    sampleIds: IndexedSeq[String] = sampleIds,
    sparseCutoff: Double = sparseCutoff,
    blockWidth: Int = blockWidth): HardCallSet =
    new HardCallSet(df, localSamples, sampleIds, sparseCutoff, blockWidth)

  def cache(): HardCallSet = copy(df = df.cache())

  def persist(level: StorageLevel): HardCallSet = copy(df = df.persist(level))

  def repartition(nPartitions: Int) = copy(df = df.repartition(nPartitions))

  /*
  def filterVariants(p: (Variant) => Boolean): HardCallSet =
    copy(rdd = rdd.filter { case (v, cs) => p(v) })
  */

  def nSamples = localSamples.size

  def nVariants: Long = rdd.count()

  def nSparseVariants: Long = rdd.filter { case (v, cs) => cs.isSparse }.count()

  def nDenseVariants: Long = rdd.filter { case (v, cs) => !cs.isSparse }.count()

  def variantCovData(variants: Array[Variant]): CovariateData = {

    def variantGtVector(v: Variant): breeze.linalg.Vector[Double] = {

      val vRow = df
        .filter(df("contig") === v.contig)
        .filter(df("block") === v.start / blockWidth)
        .filter(df("start") === v.start)
        .filter(df("ref") === v.ref)
        .filter(df("alt") === v.alt)
        .collect()

      if (vRow.size == 1)
        vRow(0).getCallStream(3).hardStats(nSamples).x
      else if (vRow.isEmpty)
        fatal(s"Covariate ${v.asString} does not refer to a variant in the data set.")
      else
        fatal(s"Covariate ${v.asString} refers to multiple variants in the data set.")
    }

    val covRowSample = localSamples

    if (variants.nonEmpty) {
      val covName = variants.map(_.asString)
      val data = DenseMatrix.zeros[Double](nSamples, variants.size)

      for ((v, j) <- variants.view.zipWithIndex)
        data(::, j to j) := variantGtVector(v)

      CovariateData(covRowSample, covName, Some(data))
    }
    else
      CovariateData(covRowSample, Array[String](), None)
  }

  def capVariantsPerBlock(maxPerBlock: Int): HardCallSet = {
    import df.sqlContext.implicits._

    val filtRDD = df
      .rdd
      .groupBy(r => (r.getString(4), r.getInt(5)))
      .mapValues(it => scala.util.Random.shuffle(it).take(maxPerBlock))
      .

    //val filtRdd = rdd.filter{ case (Variant(contig, pos, _, _), cs) => Set(("chr1", 1)) contains (contig, pos) }

    new HardCallSet(filtRdd.toDF(),
      localSamples,
      sampleIds,
      sparseCutoff,
      blockWidth)
  }

  def writeNVariantsPerBlock(filename: String) {
    def removeChr(s: String): String = if (s.startsWith("chr")) s.drop(3) else s

    df.groupBy("contig", "block")
      .count()
      .sort("contig", "block")
      .map(r => s"${removeChr(r.getString(0))}\t${r.getInt(1)}\t${r.getLong(2)}")
      .writeTable(filename, Some("CHR\tBLOCK\tN"))
  }
}


case class GtVectorAndStats(x: breeze.linalg.Vector[Double], xx: Double, nMiossing: Int)


object CallStream {

  def apply(gs: Iterable[Genotype], n: Int, sparseCutoff: Double): CallStream = {
    require(n >= 0) // FIXME: allowing n = 0 requires check that n != 0 in hardstats below. Right choice?

    val nHomRef = gs.count(_.isHomRef)

    if (n - nHomRef < n * sparseCutoff)
      sparseCallStream(gs, n, nHomRef)
    else
      denseCallStream(gs, n, nHomRef)
  }

  def denseCallStream(gs: Iterable[Genotype], n: Int, nHomRef: Int) =
    denseCallStreamFromGtStream(gs.map(_.gt.getOrElse(3)), n: Int, nHomRef: Int)

  def denseCallStreamFromGtStream(gts: Iterable[Int], n: Int, nHomRef: Int): CallStream = {
    var sumX = 0
    var sumXX = 0
    var nMissing = 0

    for (gt <- gts)
      (gt: @unchecked) match {
        case 0 =>
        case 1 =>
          sumX += 1
          sumXX += 1
        case 2 =>
          sumX += 2
          sumXX += 4
        case 3 =>
          nMissing += 1
      }

    val meanX = sumX.toDouble / (n - nMissing) // FIXME: deal with case of all missing

    new CallStream(
      denseByteArray(gts.toArray),
      meanX,
      sumXX + meanX * meanX * nMissing,
      nMissing,
      nHomRef,
      false)
  }

  def denseByteArray(gts: Array[Int]): Array[Byte] = {

    val a = Array.ofDim[Byte]((gts.length + 3) / 4)

    var i = 0
    var j = 0
    while (i + 3 < gts.length) {
      a(j) = (gts(i) | (gts(i + 1) << 2) | (gts(i + 2) << 4) | (gts(i + 3) << 6)).toByte
      i += 4
      j += 1
    }

    (gts.length - i: @unchecked) match {
      case 0 =>
      case 1 => a(j) = gts(i).toByte
      case 2 => a(j) = (gts(i) | (gts(i + 1) << 2)).toByte
      case 3 => a(j) = (gts(i) | (gts(i + 1) << 2) | (gts(i + 2) << 4)).toByte
    }

    a
  }

  def sparseCallStream(gs: Iterable[Genotype], n: Int, nHomRef: Int) =
    sparseCallStreamFromGtStream(gs.map(_.gt.getOrElse(3)), n: Int, nHomRef: Int)

  def sparseCallStreamFromGtStream(gts: Iterable[Int], n: Int, nHomRef: Int): CallStream = {
    val rowX = Array.ofDim[Int](n - nHomRef)
    val valX = Array.ofDim[Int](n - nHomRef)
    var sumX = 0
    var sumXX = 0
    var nMissing = 0

    var j = 0
    for ((gt, i) <- gts.view.zipWithIndex)
      (gt: @unchecked) match {
        case 0 =>
        case 1 =>
          rowX(j) = i
          valX(j) = 1
          sumX += 1
          sumXX += 1
          j += 1
        case 2 =>
          rowX(j) = i
          valX(j) = 2
          sumX += 2
          sumXX += 4
          j += 1
        case 3 =>
          rowX(j) = i
          valX(j) = 3
          nMissing += 1
          j += 1
      }

    val meanX = sumX.toDouble / (n - nMissing)

    new CallStream(
      sparseByteArray(rowX, valX),
      meanX,
      sumXX + meanX * meanX * nMissing,
      nMissing,
      nHomRef,
      true)
  }

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

    val a = new mutable.ArrayBuilder.ofByte

    def merge(rd: Int, l: Int) =
      (l: @unchecked) match {
        case 0 =>
          a += rd.toByte
        case 1 =>
          a += (rd >> 8).toByte
          a += rd.toByte
        case 2 =>
          a += (rd >> 16).toByte
          a += (rd >> 8).toByte
          a += rd.toByte
        case 3 =>
          a += (rd >> 24).toByte
          a += (rd >> 16).toByte
          a += (rd >> 8).toByte
          a += rd.toByte
      }

    var i = 0
    var r = 0

    while (i + 3 < gts.length) {
      val gt1 = gts(i)
      val rd1 = rows(i) - r
      val l1 = nBytesMinus1(rd1)
      r = rows(i)
      i += 1

      val gt2 = gts(i)
      val rd2 = rows(i) - r
      val l2 = nBytesMinus1(rd2)
      r = rows(i)
      i += 1

      val gt3 = gts(i)
      val rd3 = rows(i) - r
      val l3 = nBytesMinus1(rd3)
      r = rows(i)
      i += 1

      val gt4 = gts(i)
      val rd4 = rows(i) - r
      val l4 = nBytesMinus1(rd4)
      r = rows(i)
      i += 1

      a += ((gt4 << 6) | (gt3 << 4) | (gt2 << 2) | gt1).toByte // gtByte
      a += ((l4 << 6) | (l3 << 4) | (l2 << 2) | l1).toByte // gtByte
      merge(rd1, l1)
      merge(rd2, l2)
      merge(rd3, l3)
      merge(rd4, l4)
    }

    (gts.length - i: @unchecked) match {
      case 0 =>

      case 1 =>
        val gt1 = gts(i)
        val rd1 = rows(i) - r
        val l1 = nBytesMinus1(rd1)

        a += gt1.toByte
        a += l1.toByte
        merge(rd1, l1)

      case 2 =>
        val gt1 = gts(i)
        val rd1 = rows(i) - r
        val l1 = nBytesMinus1(rd1)
        r = rows(i)
        i += 1

        val gt2 = gts(i)
        val rd2 = rows(i) - r
        val l2 = nBytesMinus1(rd2)

        a += ((gt2 << 2) | gt1).toByte
        a += ((l2 << 2) | l1).toByte
        merge(rd1, l1)
        merge(rd2, l2)

      case 3 =>
        val gt1 = gts(i)
        val rd1 = rows(i) - r
        val l1 = nBytesMinus1(rd1)
        r = rows(i)
        i += 1

        val gt2 = gts(i)
        val rd2 = rows(i) - r
        val l2 = nBytesMinus1(rd2)
        r = rows(i)
        i += 1

        val gt3 = gts(i)
        val rd3 = rows(i) - r
        val l3 = nBytesMinus1(rd3)

        a += ((gt3 << 4) | (gt2 << 2) | gt1).toByte
        a += ((l3 << 4) | (l2 << 2) | l1).toByte
        merge(rd1, l1)
        merge(rd2, l2)
        merge(rd3, l3)
    }

    a.result()
  }

  def toBinaryString(b: Byte): String = {for (i <- 7 to 0 by -1) yield (b & (1 << i)) >> i}.mkString("")

  def toIntsString(b: Byte): String = {for (i <- 6 to 0 by -2) yield (b & (3 << i)) >> i}.mkString(":")
}

case class CallStream(a: Array[Byte], meanX: Double, sumXX: Double, nMissing: Int, nHomRef: Int, isSparse: Boolean) {

  def hardStats(n: Int): GtVectorAndStats = {
    if (n == 0)
      fatal("Cannot compute statistics for 0 samples.")
    if (isSparse)
      sparseStats(n)
    else
      denseStats(n)
  }

  def denseStats(n: Int): GtVectorAndStats = {

    val x = Array.ofDim[Double](n)

    val mask00000011 = 3
    val mask00001100 = 3 << 2
    val mask00110000 = 3 << 4
    val mask11000000 = 3 << 6

    def merge(i: Int, gt: Int) {
      (gt: @unchecked) match {
        case 0 =>
        case 1 =>
          x(i) = 1.0
        case 2 =>
          x(i) = 2.0
        case 3 =>
          x(i) = this.meanX
      }
    }

    var i = 0
    var j = 0

    while (i < n - 3) {
      val b = a(j)
      merge(i, b & mask00000011)
      merge(i + 1, (b & mask00001100) >> 2)
      merge(i + 2, (b & mask00110000) >> 4)
      merge(i + 3, (b & mask11000000) >> 6)

      i += 4
      j += 1
    }

    (n - i: @unchecked) match {
      case 0 =>
      case 1 =>
        merge(i, a(j) & mask00000011)
      case 2 =>
        merge(i, a(j) & mask00000011)
        merge(i + 1, (a(j) & mask00001100) >> 2)
      case 3 =>
        merge(i, a(j) & mask00000011)
        merge(i + 1, (a(j) & mask00001100) >> 2)
        merge(i + 2, (a(j) & mask00110000) >> 4)
    }

    GtVectorAndStats(DenseVector(x), sumXX, nMissing)
  }

  def sparseStats(n: Int): GtVectorAndStats = {

    val rowX = Array.ofDim[Int](n - nHomRef)
    val valX = Array.ofDim[Double](n - nHomRef)

    val mask00000011 = 3
    val mask00001100 = 3 << 2
    val mask00110000 = 3 << 4
    val mask11000000 = 3 << 6

    def merge(i: Int, r: Int, gt: Int) {
      (gt: @unchecked) match {
        case 0 =>
        case 1 =>
          rowX(i) = r
          valX(i) = 1.0
        case 2 =>
          rowX(i) = r
          valX(i) = 2.0
        case 3 =>
          rowX(i) = r
          valX(i) = meanX
      }
    }

    // FIXME: Can we somehow store with & 0xFF already applied on encode so that we don't need it on decode?
    def rowDiff(k: Int, l: Int) =
      (l: @unchecked) match {
        case 0 => a(k) & 0xFF
        case 1 => ((a(k) & 0xFF) << 8) | (a(k + 1) & 0xFF)
        case 2 => ((a(k) & 0xFF) << 16) | ((a(k + 1) & 0xFF) << 8) | (a(k + 2) & 0xFF)
        case 3 => (a(k) << 24) | ((a(k + 1) & 0xFF) << 16) | ((a(k + 2) & 0xFF) << 8) | (a(k + 3) & 0xFF)
      }

    var i = 0 // row index
    var j = 0 // byte index
    var r = 0 // row

    while (i + 3 < n - nHomRef) {
      val gtByte = a(j)
      val gt1 = gtByte & mask00000011
      val gt2 = (gtByte & mask00001100) >> 2
      val gt3 = (gtByte & mask00110000) >> 4
      val gt4 = (gtByte & mask11000000) >> 6

      val lenByte = a(j + 1)
      val l1 = lenByte & mask00000011
      val l2 = (lenByte & mask00001100) >> 2
      val l3 = (lenByte & mask00110000) >> 4
      val l4 = (lenByte & mask11000000) >> 6

      j += 2

      r += rowDiff(j, l1)
      merge(i, r, gt1)
      i += 1
      j += l1 + 1

      r += rowDiff(j, l2)
      merge(i, r, gt2)
      i += 1
      j += l2 + 1

      r += rowDiff(j, l3)
      merge(i, r, gt3)
      i += 1
      j += l3 + 1

      r += rowDiff(j, l4)
      merge(i, r, gt4)
      i += 1
      j += l4 + 1
    }

    (n - nHomRef - i: @unchecked) match {
      case 0 =>

      case 1 =>
        val gtByte = a(j)
        val gt1 = gtByte & mask00000011

        val lenByte = a(j + 1)
        val l1 = lenByte & mask00000011

        j += 2

        r += rowDiff(j, l1)
        merge(i, r, gt1)

      case 2 =>
        val gtByte = a(j)
        val gt1 = gtByte & mask00000011
        val gt2 = (gtByte & mask00001100) >> 2

        val lenByte = a(j + 1)
        val l1 = lenByte & mask00000011
        val l2 = (lenByte & mask00001100) >> 2

        j += 2

        r += rowDiff(j, l1)
        merge(i, r, gt1)
        i += 1
        j += l1 + 1

        r += rowDiff(j, l2)
        merge(i, r, gt2)

      case 3 =>
        val gtByte = a(j)
        val gt1 = gtByte & mask00000011
        val gt2 = (gtByte & mask00001100) >> 2
        val gt3 = (gtByte & mask00110000) >> 4

        val lenByte = a(j + 1)
        val l1 = a(j + 1) & mask00000011
        val l2 = (a(j + 1) & mask00001100) >> 2
        val l3 = (a(j + 1) & mask00110000) >> 4

        j += 2

        r += rowDiff(j, l1)
        merge(i, r, gt1)
        i += 1
        j += l1 + 1

        r += rowDiff(j, l2)
        merge(i, r, gt2)
        i += 1
        j += l2 + 1

        r += rowDiff(j, l3)
        merge(i, r, gt3)
    }

    GtVectorAndStats(new SparseVector[Double](rowX, valX, n), sumXX, nMissing)
  }

  import CallStream._

  def showBinary() = println(a.map(b => toBinaryString(b)).mkString("[", ", ", "]"))

  override def toString = s"${a.map(b => toBinaryString(b)).mkString("[", ", ", "]")}, $meanX, $sumXX, $nMissing"
}