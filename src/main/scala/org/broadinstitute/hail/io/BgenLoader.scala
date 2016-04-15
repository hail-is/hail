package org.broadinstitute.hail.io

import org.apache.hadoop.io.LongWritable
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant._

import scala.io.Source

case class BgenState(compressed: Boolean, nSamples: Int, nVariants: Int,
  headerLength: Int, dataStart: Int, hasIds: Boolean)

case class BgenResult(file: String, nSamples: Int, nVariants: Int, rdd: RDD[(Variant, Annotation, Iterable[Genotype])])

object BgenLoader {
  final val MAX_PL = 51

  lazy val phredConversionTable: Array[Double] = (0 to 65535).map { i => if (i == 0) MAX_PL else -10 * math.log10(i) }
    .toArray

  def load(sc: SparkContext, file: String, nPartitions: Option[Int] = None): BgenResult = {

    val bState = readState(sc.hadoopConfiguration, file)

    BgenResult(file, bState.nSamples, bState.nVariants,
      sc.hadoopFile(file, classOf[BgenInputFormat], classOf[LongWritable], classOf[ParsedLine[Variant]],
      nPartitions.getOrElse(sc.defaultMinPartitions))
      .map { case (lw, pl) => (pl.getKey, pl.getAnnotation, pl.getGS) })
  }

  def index(hConf: org.apache.hadoop.conf.Configuration, file: String) {
    val indexFile = file + ".idx"

    val bState = readState(hConf, file)

    val dataBlockStarts = new Array[Long](bState.nVariants + 1)
    var position: Long = bState.dataStart

    dataBlockStarts(0) = position

    readFile(file, hConf) { is =>
      val reader = new HadoopFSDataBinaryReader(is)
      reader.seek(0)

      for (i <- 1 until bState.nVariants + 1) {
        reader.seek(position)

        val nRow = reader.readInt()

        val snpid = reader.readLengthAndString(2)
        val rsid = reader.readLengthAndString(2)
        val chr = reader.readLengthAndString(2)
        val pos = reader.readInt()

        reader.readLengthAndString(4) // read an allele
        reader.readLengthAndString(4) // read an allele


        position = if (bState.compressed)
          reader.readInt() + reader.getPosition
        else
          reader.getPosition + 6 * bState.nSamples

        dataBlockStarts(i) = position
      }
    }

    IndexBTree.write(dataBlockStarts, indexFile, hConf)

  }

  def readSamples(hConf: org.apache.hadoop.conf.Configuration, file: String): IndexedSeq[String] = {
    val bState = readState(hConf, file)
    if (!bState.hasIds)
      fatal(s"BGEN file `$file' contains no sample ID block, coimport a `.sample' file")

    readFile(file, hConf) { is =>
      val reader = new HadoopFSDataBinaryReader(is)

      reader.seek(bState.headerLength + 4)
      val sampleIdSize = reader.readInt()
      val nSamples = reader.readInt()

      if (nSamples != bState.nSamples)
        fatal("BGEN file is malformed -- number of sample IDs in header does not equal number in file")

      if (sampleIdSize + bState.headerLength > bState.dataStart - 4)
        fatal("BGEN file is malformed -- offset is smaller than length of header")

      (0 until nSamples).map { i =>
        reader.readLengthAndString(2)
      }
    }
  }

  def readSampleFile(hConf: org.apache.hadoop.conf.Configuration, file: String): IndexedSeq[String] = {
    readFile(file, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .drop(2)
        .filter(line => !line.isEmpty)
        .map { line =>
          val arr = line.split("\\s+")
          arr(0)
        }
        .toIndexedSeq
    }
  }

  def readState(hConf: org.apache.hadoop.conf.Configuration, file: String): BgenState = {
    readFile(file, hConf) { is =>
      val reader = new HadoopFSDataBinaryReader(is)
      readState(reader)
    }
  }

  def readState(reader: HadoopFSDataBinaryReader): BgenState = {
    reader.seek(0)
    val allInfoLength = reader.readInt()
    val headerLength = reader.readInt()
    val dataStart = allInfoLength + 4

    assert(headerLength <= allInfoLength)
    val nVariants = reader.readInt()
    val nSamples = reader.readInt()

    val magicNumber = reader.readBytes(4)
      .map(_.toInt)
      .toSeq

    if (magicNumber != Seq(0, 0, 0, 0) && magicNumber != Seq(98, 103, 101, 110))
      fatal(s"expected magic number [0000] or [bgen], got [${magicNumber.mkString}]")

    if (headerLength > 20)
      reader.skipBytes(headerLength.toInt - 20)

    val flags = reader.readInt()
    val compression = (flags & 1) != 0
    val version = flags >> 2 & 0xf
    if (version != 1)
      fatal(s"Hail supports BGEN version 1.1, got version 1.$version")

    val hasIds = (flags >> 30 & 1) != 0
    BgenState(compression, nSamples, nVariants, headerLength, dataStart, hasIds)
  }
}
