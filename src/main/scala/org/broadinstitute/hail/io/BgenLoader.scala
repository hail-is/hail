package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.apache.hadoop.io.LongWritable
import org.apache.spark.SparkContext

import scala.io.Source

class BgenLoader(file: String, sampleFile: Option[String] = None, sc: SparkContext) {
  private var compression: Boolean = false
  private var version: Int = 0
  private var hasSampleIdBlock: Boolean = false
  private val reader = new HadoopFSDataBinaryReader(hadoopOpen(file, sc.hadoopConfiguration))
  private var nSamples: Int = 0
  private var nVariants: Int = 0
  private var dataStart: Int = 24
  private var headerLength: Int = 0
  private var allInfoLength: Int = 0
  parseHeader()
  val sampleIds: Array[String] = parseSampleIds(sampleFile)

  def getSampleIds: Array[String] = sampleIds
  def getNSamples: Int = nSamples
  def getNVariants: Int = nVariants
  def getIndex: String = file + ".idx"
  def getFile: String = file

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

  def createIndex() = {
    val indexFile = file + ".idx"
    val dataBlockStarts = new Array[Long](nVariants + 1)
    var position: Long = dataStart

    dataBlockStarts(0) = position

    for (i <- 1 until nVariants + 1) {
      position = getNextBlockPosition(position)
      dataBlockStarts(i) = position
      if (i % 100000 == 0)
        info(s"Read the ${i}th variant out of $nVariants in file [$file]")
    }

    IndexBTree.write(dataBlockStarts, indexFile, sc.hadoopConfiguration)
  }

  def parseSampleIds(sampleFile: Option[String] = None): Array[String] = {
    if (!hasSampleIdBlock && sampleFile.isEmpty)
      fatal("No sample ids detected in BGEN file. Use -s with Sample ID file")
    else if (hasSampleIdBlock && sampleFile.isDefined)
      warn("Sample ids detected in BGEN file but Sample ID file given. Using IDs from sample ID file")

    val sampleIDs = {
      if (sampleFile.isDefined)
        BgenLoader.parseSampleFile(sampleFile.get, sc.hadoopConfiguration)
      else {
        reader.seek(headerLength + 4)
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
      fatal(s"Length of sample IDs does not equal number of samples in BGEN file (${sampleIDs.length}, $nSamples)")

    sampleIDs
  }

  private def parseHeader() = {
    reader.seek(0)
    allInfoLength = reader.readInt()
    headerLength = reader.readInt()

    // allInfoLength is the "offset relative to the 5th byte of the start of the first variant block
    dataStart = allInfoLength + 4

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

  def createIndex(bgenFile: String, sampleFile: Option[String] = None, sc: SparkContext) = {
    val bl = new BgenLoader(bgenFile, sampleFile, sc)
    info(s"Creating index file at ${bgenFile + ".idx"}")
    bl.createIndex()
  }

  def readData(bgenFiles: Array[String], sampleFile: Option[String] = None, sc: SparkContext,
            nPartitions: Option[Int] = None, compress: Boolean = true): VariantDataset = {

    val bgenLoaders = bgenFiles.map{file => new BgenLoader(file, sampleFile, sc)}

    val bgenIndexed = bgenFiles.forall{file =>
      if (!hadoopIsFile(file + ".idx", sc.hadoopConfiguration)) {
        warn(s"No index file detected for $file")
        false
      }
      else
        true
    }

    if (!bgenIndexed)
      fatal(s"Not all BGEN files have been indexed. Run the command indexbgen.")

    val sampleBgen = bgenLoaders(0)

    val nSamplesEqual = bgenLoaders.map{_.getNSamples}forall(_.equals(sampleBgen.getNSamples))
    if (!nSamplesEqual)
      fatal("Different number of samples in BGEN files")

    val sampleIDsEqual = bgenLoaders.map{_.getSampleIds}.forall(_.sameElements(sampleBgen.getSampleIds))
    if (!sampleIDsEqual)
      fatal("Sample IDs are not equal across BGEN files")


    val nSamples = sampleBgen.getNSamples
    val nVariants = bgenLoaders.map{_.getNVariants}.sum

    if (!bgenLoaders.forall(_.getNVariants > 0))
      fatal("Require at least 1 Variant in each BGEN file")

    val sampleIDs = sampleBgen.getSampleIds

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