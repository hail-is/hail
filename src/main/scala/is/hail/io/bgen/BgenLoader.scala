package is.hail.io.bgen

import org.apache.hadoop.io.LongWritable
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.annotations._
import is.hail.driver.HailContext
import is.hail.expr._
import is.hail.io._
import is.hail.io.gen.GenReport
import is.hail.variant._

import scala.collection.mutable
import scala.io.Source

case class BgenHeader(compressed: Boolean, nSamples: Int, nVariants: Int,
  headerLength: Int, dataStart: Int, hasIds: Boolean)

case class BgenResult(file: String, nSamples: Int, nVariants: Int, rdd: RDD[(LongWritable, BgenRecord)])

object BgenLoader {

  def load(hc: HailContext, files: Array[String], sampleFile: Option[String] = None,
    tolerance: Double, compress: Boolean, nPartitions: Option[Int] = None): VariantDataset = {
    require(files.nonEmpty)
    val samples = sampleFile.map(file => BgenLoader.readSampleFile(hc.hadoopConf, file))
      .getOrElse(BgenLoader.readSamples(hc.hadoopConf, files.head))

    val duplicateIds = samples.duplicates().toArray
    if (duplicateIds.nonEmpty) {
      val n = duplicateIds.length
      warn(s"""found $n duplicate sample ${plural(n, "ID")}
               |  Duplicate IDs: @1""".stripMargin, duplicateIds)
    }

    val nSamples = samples.length

    hc.hadoopConf.setBoolean("compressGS", compress)
    hc.hadoopConf.setDouble("tolerance", tolerance)

    val sc = hc.sc
    val results = files.map { file =>
      val reportAcc = sc.accumulable[mutable.Map[Int, Int], Int](mutable.Map.empty[Int, Int])
      val bState = readState(sc.hadoopConfiguration, file)
      GenReport.accumulators ::= (file, reportAcc)
      BgenResult(file, bState.nSamples, bState.nVariants,
        sc.hadoopFile(file, classOf[BgenInputFormat], classOf[LongWritable], classOf[BgenRecord],
          nPartitions.getOrElse(sc.defaultMinPartitions)))
    }

    val unequalSamples = results.filter(_.nSamples != nSamples).map(x => (x.file, x.nSamples))
    if (unequalSamples.length > 0)
      fatal(
        s"""The following BGEN files did not contain the expected number of samples $nSamples:
            |  ${ unequalSamples.map(x => s"""(${ x._2 } ${ x._1 }""").mkString("\n  ") }""".stripMargin)

    val noVariants = results.filter(_.nVariants == 0).map(_.file)
    if (noVariants.length > 0)
      fatal(
        s"""The following BGEN files did not contain at least 1 variant:
            |  ${ noVariants.mkString("\n  ") })""".stripMargin)

    val nVariants = results.map(_.nVariants).sum

    info(s"Number of BGEN files parsed: ${ results.length }")
    info(s"Number of samples in BGEN files: $nSamples")
    info(s"Number of variants across all BGEN files: $nVariants")

    val signature = TStruct("rsid" -> TString, "varid" -> TString)

    val fastKeys = sc.union(results.map(_.rdd.map(_._2.getKey)))

    val rdd = sc.union(results.map(_.rdd.map { case (_, decoder) =>
      (decoder.getKey, (decoder.getAnnotation, decoder.getValue))
    })).toOrderedRDD[Locus](fastKeys)

    VariantSampleMatrix(hc, VariantMetadata(
      sampleIds = samples,
      sampleAnnotations = IndexedSeq.fill(nSamples)(Annotation.empty),
      globalAnnotation = Annotation.empty,
      saSignature = TStruct.empty,
      vaSignature = signature,
      globalSignature = TStruct.empty,
      wasSplit = true,
      isDosage = true), rdd)
  }

  def index(hConf: org.apache.hadoop.conf.Configuration, file: String) {
    val indexFile = file + ".idx"

    val bState = readState(hConf, file)

    val dataBlockStarts = new Array[Long](bState.nVariants + 1)
    var position: Long = bState.dataStart

    dataBlockStarts(0) = position

    hConf.readFile(file) { is =>
      val reader = new HadoopFSDataBinaryReader(is)
      reader.seek(0)

      for (i <- 1 to bState.nVariants) {
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

  def readSamples(hConf: org.apache.hadoop.conf.Configuration, file: String): Array[String] = {
    val bState = readState(hConf, file)
    if (!bState.hasIds)
      fatal(s"BGEN file `$file' contains no sample ID block, coimport a `.sample' file")

    hConf.readFile(file) { is =>
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
      }.toArray
    }
  }

  def readSampleFile(hConf: org.apache.hadoop.conf.Configuration, file: String): Array[String] = {
    hConf.readFile(file) { s =>
      Source.fromInputStream(s)
        .getLines()
        .drop(2)
        .filter(line => !line.isEmpty)
        .map { line =>
          val arr = line.split("\\s+")
          arr(0)
        }
        .toArray
    }
  }

  def readState(hConf: org.apache.hadoop.conf.Configuration, file: String): BgenHeader = {
    hConf.readFile(file) { is =>
      val reader = new HadoopFSDataBinaryReader(is)
      readState(reader)
    }
  }

  def readState(reader: HadoopFSDataBinaryReader): BgenHeader = {
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
      fatal(s"expected magic number [0000] or [bgen], got [${ magicNumber.mkString }]")

    if (headerLength > 20)
      reader.skipBytes(headerLength.toInt - 20)

    val flags = reader.readInt()
    val compression = (flags & 1) != 0
    val version = flags >> 2 & 0xf
    if (version != 1)
      fatal(s"Hail supports BGEN version 1.1, got version 1.$version")

    val hasIds = (flags >> 30 & 1) != 0
    BgenHeader(compression, nSamples, nVariants, headerLength, dataStart, hasIds)
  }
}
