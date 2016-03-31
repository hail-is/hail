package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.apache.hadoop.io.LongWritable
import org.apache.spark.SparkContext

import scala.io.Source

class BgenLoader(file: String, sampleFile: Option[String], sc: SparkContext) {
  private var compression: Boolean = false
  private var version: Int = 0
  private var hasSampleIdBlock: Boolean = false
  private val reader = new HadoopFSDataBinaryReader(hadoopOpen(file, sc.hadoopConfiguration))
  private var nSamples: Int = 0
  private var nVariants: Int = 0
  private var sampleIDs: Array[String] = parseHeaderAndIndex(sampleFile)

  def getNSamples: Int = nSamples
  def getNVariants: Int = nVariants
  def getSampleIDs: Array[String] = sampleIDs

  private def getNextBlockPosition(position: Long): Long = {
    reader.seek(position)

    var nRow = nSamples //check this gets updated properly
    if (version == 1)
      nRow = reader.readInt()

    val snpid = reader.readLengthAndString(2)
    val rsid = reader.readLengthAndString(2)
    val chr = reader.readLengthAndString(2)
    val pos = reader.readInt()

    val nAlleles = if (version == 1) 2 else reader.readShort()

    for (i <- 0 until nAlleles)
      reader.readLengthAndString(4) // read an allele

    // Read the size of the genotype probability block
    if (version == 1 && !compression)
      reader.getPosition + 6 * nRow
    else if (version == 1 && compression)
      reader.readInt() + reader.getPosition
    else
      throw new UnsupportedOperationException()
  }

  def parseHeaderAndIndex(sampleFile: Option[String] = None): Array[String] = {
    reader.seek(0)
    val allInfoLength = reader.readInt()
    val headerLength = reader.readInt()

    require(headerLength <= allInfoLength)

    nVariants = reader.readInt()
    nSamples = reader.readInt()
    val magicNumber = reader.readString(4) //readers ignore these bytes

    val headerInfo = {
      if (headerLength > 20)
        reader.readString(headerLength.toInt - 20)
      else
        ""
    }

    val flags = reader.readInt()
    compression = (flags & 1) != 0 // either 0 or 1 based on the first bit
    version = flags >> 2 & 0xf
    hasSampleIdBlock = (flags >> 30 & 1) != 0

    if (version != 1) // FIXME add support for more than 1 bit for v1.2
      fatal("Hail supports only BGEN v1.1 formats")

    if (!hasSampleIdBlock && sampleFile.isEmpty)
      fatal("No sample ids detected in BGEN file. Use -s with Sample ID file")
    else if (hasSampleIdBlock && sampleFile.isDefined)
      warn("Sample ids detected in BGEN file but Sample ID file given. Using IDs from sample ID file")

    val sampleIDs = {
      if (sampleFile.isDefined)
        BgenLoader.parseSampleFile(sampleFile.get, sc.hadoopConfiguration)
      else {
        val sampleIdSize = reader.readInt()
        val nSamplesConfirmation = reader.readInt()

        if (nSamplesConfirmation != nSamples)
          fatal("BGEN file is malformed -- number of sample IDs in header does not equal number in file")

        if (sampleIdSize + headerLength > allInfoLength)
          fatal("BGEN file is malformed -- offset is smaller than length of header")

        val sampleIdArr = new Array[String](nSamples)
        for (i <- 0 until nSamples) {
          sampleIdArr(i) = reader.readLengthAndString(2)
        }
        sampleIdArr
      }
    }

    if (sampleIDs.length != nSamples)
      fatal(s"Length of sample IDs does not equal number of samples in BGEN file (${sampleIDs.length}, ${nSamples})")

    if (!hadoopIsFile(file + ".idx", sc.hadoopConfiguration)) {
      info(s"Creating index [${file + ".idx"}]")
      val dataBlockStarts = new Array[Long](nVariants + 1)

      // allInfoLength is the "offset relative to the 5th byte of the start of the first variant block
      var position: Long = (allInfoLength + 4).toLong
      dataBlockStarts(0) = position

      for (i <- 1 until nVariants + 1) {
        position = getNextBlockPosition(position)
        dataBlockStarts(i) = position
      }

      IndexBTree.write(dataBlockStarts, file + ".idx", sc.hadoopConfiguration)
    }

    sampleIDs
  }
}

object BgenLoader {

  def parseGenotype(pls: Array[Int]): Int = {
    if (pls(0) == 0 && pls(1) == 0
      || pls(0) == 0 && pls(2) == 0
      || pls(1) == 0 && pls(2) == 0)
      -1
    else {
      if (pls(0) == 0)
        0
      else if (pls(1) == 0)
        1
      else if (pls(2) == 0)
        2
      else
        -1
    }
  }

  val phredConversionTable: Array[Double] = (0 to 65535).map{i => if (i == 0) 48 else -10 * math.log10(i)}.toArray

  def phredScalePPs(probAA: Int, probAB: Int, probBB: Int): Array[Int] = {
    if (probAA == 32768 || probBB == 32768 || probAB == 32768) {
      Array(if (probAA == 32768) 0 else 48, if (probAB == 32768) 0 else 48, if (probBB == 32768) 0 else 48)
    }
    else {
      val phredDoubles: (Double, Double, Double) = (
        if (probAA == 0) 48 else phredConversionTable(probAA),
        if (probAB == 0) 48 else phredConversionTable(probAB),
        if (probBB == 0) 48 else phredConversionTable(probBB))

      val minValue = math.min(math.min(phredDoubles._1, phredDoubles._2), phredDoubles._3)
      Array((phredDoubles._1 - minValue + .5).toInt,
        (phredDoubles._2 - minValue + .5).toInt,
        (phredDoubles._3 - minValue + .5).toInt)
    }
  }

  def parseSampleFile(file: String, hConf: Configuration): Array[String] = {
    readFile(file, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .drop(2)
        .filter(line => !line.isEmpty)
        .map { line =>
          val arr = line.split("\\s+")
          arr(0) // using ID_1
        }
        .toArray
    }
  }

  def apply(bgenFiles: Array[String], sampleFile: Option[String] = None, sc: SparkContext,
            nPartitions: Option[Int] = None, compress: Boolean = true): VariantDataset = {
    val bgenLoaders = bgenFiles.map{file => new BgenLoader(file, sampleFile, sc)}
    val nSamplesEqual = bgenLoaders.map{_.getNSamples}.forall(_.equals(bgenLoaders(0).getNSamples))
    if (!nSamplesEqual)
      fatal("Different number of samples in BGEN files")

    val sampleIDsEqual = bgenLoaders.map{_.getSampleIDs}.forall(_.sameElements(bgenLoaders(0).getSampleIDs))
    if (!sampleIDsEqual)
      fatal("Sample IDs are not equal across BGEN files")

    val nSamples = bgenLoaders(0).getNSamples
    val nVariants = bgenLoaders.map{_.getNVariants}.sum

    if (nVariants < 1)
      fatal("Require at least 1 Variant in BGEN files")

    val sampleIDs = bgenLoaders(0).getSampleIDs

    info(s"Number of BGEN files parsed: ${bgenLoaders.length}")
    info(s"Number of variants in all BGEN files: $nVariants")
    info(s"Number of samples in BGEN files: $nSamples")

    sc.hadoopConfiguration.setBoolean("compressGS", compress)

    val signatures = Annotations(Map("rsid" -> new SimpleSignature("String"), "varid" -> new SimpleSignature("String")))

    VariantSampleMatrix(metadata = VariantMetadata(sampleIDs).addVariantAnnotationSignatures(signatures), rdd = sc.union(bgenFiles.map{ file =>
      sc.hadoopFile(file, classOf[BgenInputFormat], classOf[LongWritable], classOf[ParsedLine[Variant]],
        nPartitions.getOrElse(sc.defaultMinPartitions))
        .map { case (lw, pl) => (pl.getKey, pl.getAnnotation, pl.getGS) }
    }))
  }
}