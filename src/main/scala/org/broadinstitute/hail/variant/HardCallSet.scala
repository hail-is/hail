package org.broadinstitute.hail.variant

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._

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

    val nSamples = readDataFile(dirname + "/metadata.ser",
      sqlContext.sparkContext.hadoopConfiguration) { dis => readData[Int](dis) }

    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    new HardCallSet(df.rdd.map(r => (r.getVariant(0), r.getAs[CallStream](1))), nSamples)

  }
}

case class HardCallSet(rdd: RDD[(Variant, CallStream)], nSamples: Int) {
  def write(sqlContext: SQLContext, dirname: String) {
    require(dirname.endsWith(".hcs"))

    import sqlContext.implicits._

    val hConf = rdd.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeDataFile(dirname + "/metadata.ser", hConf) { dos => writeData[Int](dos, nSamples) }

    rdd.toDF().write.parquet(dirname + "/rdd.parquet")
  }
}

abstract class CallStream {
//  def toLinRegBuilder:

  type CallStream = Iterable[(Int, Int)]

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

object DenseCallStream {

  def apply(gts: Array[Int]): DenseCallStream = {

    val a = Array.ofDim[Byte]((gts.length + 3) / 4)

    var i = 0
    var s = 0
    while (s < gts.length - 4) {
      a(i) = (gts(s) | gts(s + 1) << 2 | gts(s + 2) << 4 | gts(s + 3) << 6).toByte
      i += 1
      s += 4
    }

    if (gts.length != s)
      a(i) = (
        if (gts.length == s + 1)
          gts(s)
        else if (gts.length == s + 2)
          gts(s) | gts(s + 1) << 2
        else
          gts(s) | gts(s + 1) << 2 | gts(s + 2) << 4
        ).toByte

    DenseCallStream(a)
  }
}

case class DenseCallStream(a: Array[Byte]) extends CallStream {

  def showBinary() = println(a.map(b => toBinaryString(b)).mkString("[", ", ", "]"))

  override def toString = a.map(b => toIntsString(b)).mkString("[", ", ", "]")
}

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