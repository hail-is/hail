package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.broadinstitute.hail.annotations.Annotations
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
  var nSamples: Int = 0
  var nVariants: Int = 0
  private var sampleIDs: Array[String] = parseHeaderAndIndex(sampleFile)

  def getNSamples: Int = nSamples
  def getNVariants: Int = nVariants
  def getSampleIDs: Array[String] = sampleIDs


  private def getNextBlockPosition(position: Long): Long = {
    // First seek to the proper location
    reader.seek(position)

    // Read through the variant ID block
    var nRow = nSamples
    if (version == 1)
      nRow = reader.readInt()

    val snpid = reader.readLengthAndString(2) // Lid
    val rsid = reader.readLengthAndString(2) // rsid
    val chr = reader.readLengthAndString(2) // chr
    val pos = reader.readInt() // pos

    val nAlleles = if (version == 1) 2 else reader.readShort()

    for (i <- 0 until nAlleles)
      reader.readLengthAndString(4) // read an allele

    // Read the size of the genotype probability block
    if (version == 1 && !compression)
      reader.getPosition + 6 * nRow
    else if (version == 1 && compression) {
      // The following works for 1.1 compressed
      reader.readInt() + reader.getPosition
    }
    else
      throw new UnsupportedOperationException()
  }

  def parseHeaderAndIndex(sampleFile: Option[String] = None): Array[String] = {
    // read the length of all header stuff
    reader.seek(0)
    val allInfoLength = reader.readInt()
    println(s"parseHeaderAndIndex::allInfoLength=$allInfoLength")
    // read the header block
    val headerLength = reader.readInt()

    require(headerLength <= allInfoLength)

    println(s"parseHeaderAndIndex::headerLength=$headerLength")
//    println("header len is " + headerLength)
    nVariants = reader.readInt()

    println(s"parseHeaderAndIndex::nVariants=$nVariants")
    //println("nVariants is " + nVariants)
    nSamples = reader.readInt()
    //println("nSamples is " + nSamples)
    println(s"parseHeaderAndIndex::nSamples=$nSamples")
    val magicNumber = reader.readString(4) //readers ignore these bytes
    println(s"parseHeaderAndIndex::magicNumber=$magicNumber")

    val headerInfo = {
      if (headerLength > 20)
        reader.readString(headerLength.toInt - 20)
      else
        ""
    }
    println(s"parseHeaderAndIndex::headerInfo=$headerInfo")

    // Parse flags
    val flags = reader.readInt()
    println(s"parseHeaderAndIndex::flags=$flags")
    compression = (flags & 1) != 0 // either 0 or 1 based on the first bit
    version = flags >> 2 & 0xf
    hasSampleIdBlock = (flags >> 30 & 1) != 0
    println(s"parseHeaderAndIndex compression=$compression version=$version hasSampleIdBlock=$hasSampleIdBlock")

    // version 1.1 is currently supported
    if (version != 1)
      fatal("Hail supports only BGEN v1.1 formats")

    //println("flags stuff: compression=" + compression + ", version=" + version + ", hasSampleID=" + hasSampleIdBlock)

    if (!hasSampleIdBlock && sampleFile.isEmpty)
      fatal("No sample ids detected in BGEN file. Use -s with Sample ID file")
    else if (hasSampleIdBlock && sampleFile.isDefined)
      warn("Sample ids detected in BGEN file but Sample ID file given. Using IDs from sample ID file")

    val sampleIDs = {
      if (sampleFile.isDefined)
        BgenLoader.parseSampleFile(sampleFile.get, sc.hadoopConfiguration)
      else {
        val sampleIdSize = reader.readInt()
        println("sampleIdSize is " + sampleIdSize)
        val nSamplesConfirmation = reader.readInt()
        println("nSamplesConfirm=" + nSamplesConfirmation)
        assert(nSamplesConfirmation == nSamples)

        val sampleIdArr = new Array[String](nSamples)
        for (i <- 0 until nSamples) {
          sampleIdArr(i) = reader.readLengthAndString(2)
        }
        sampleIdArr
      }
    }

    if (sampleIDs.length != nSamples)
      fatal(s"Length of sample IDs in file [$sampleFile] does not equal number of samples in BGEN file $file")

    // Read the beginnings of each variant data block
    val dataBlockStarts = new Array[Long](nVariants+1)

    // allInfoLength is the "offset relative to the 5th byte of the start of the first variant block
    var position: Long = (allInfoLength + 4).toLong
    dataBlockStarts(0) = position
    var time = System.currentTimeMillis()


    for (i <- 1 until nVariants+1) {
      position = getNextBlockPosition(position)
      dataBlockStarts(i) = position
    }

    println(s"first 3 = ${dataBlockStarts.take(3).mkString(",")}")
    println(s"last 3 = ${dataBlockStarts.takeRight(3).mkString(",")}")

    IndexBTree.write(dataBlockStarts, hadoopCreate(file + ".idx", sc.hadoopConfiguration))

    sampleIDs
  }
}

object BgenLoader {

  def parseGenotype(pls: Array[Int]): Int = {
    if (pls(0) == 0 && pls(1) == 0
      || pls(0) == 0 && pls(2) == 0
      || pls(0) == 0 && pls(2) == 0)
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

  def phredScalePPs(probAA: Int, probAB: Int, probBB: Int): Array[Int] = {
    if (probAA == 32768 || probBB == 32768 || probAB == 32768) {
      Array(if (probAA == 32768) 0 else 48, if (probAB == 32768) 0 else 48, if (probBB == 32768) 0 else 48)
    }
    else {
      val phredDoubles: (Double, Double, Double) = (
        if (probAA == 0) 48 else -10 * math.log10(probAA),
        if (probAB == 0) 48 else  -10 * math.log10(probAB),
        if (probBB == 0) 48 else  -10 * math.log10(probBB))

      val minValue = math.min(math.min(phredDoubles._1, phredDoubles._2), phredDoubles._3)
      Array((phredDoubles._1 - minValue + .5).toInt,
        (phredDoubles._2 - minValue + .5).toInt,
        (phredDoubles._3 - minValue + .5).toInt)
    }
  }

  def expectedSize(nSamples: Int, nVariants: Long): Long = {
      //this is only used for random generator testing to not have partition size be smaller than file size
      nVariants * (31 + 6*nSamples) + 24
  }

  def parseSampleFile(file: String, hConf: Configuration): Array[String] = {
    readFile(file, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .drop(2)
        .filter(line => !line.isEmpty)
        .map { line =>
          val arr = line.split("\\s+")
          arr(1)
        }
        .toArray
    }
  }

  def apply(bgenFiles: Array[String], sampleFile: Option[String] = None, sc: SparkContext,
            nPartitions: Option[Int] = None, compress: Boolean = true, gtProbThreshold: Double = 0.8): VariantDataset = {
    val bgenLoaders = bgenFiles.map{file => new BgenLoader(file, sampleFile, sc)}
    val nSamplesEqual = bgenLoaders.map{_.getNSamples}.forall(_.equals(bgenLoaders(0).getNSamples))
    if (!nSamplesEqual)
      fatal("Different number of samples in BGEN files")

    val sampleIDsEqual = bgenLoaders.map{_.getSampleIDs}.forall(_.sameElements(bgenLoaders(0).getSampleIDs))
    if (!sampleIDsEqual)
      fatal("Sample IDs are not equal across BGEN files")

    val nSamples = bgenLoaders(0).getNSamples
    val nVariants = bgenLoaders.map{_.getNVariants}.sum
    val sampleIDs = bgenLoaders(0).getSampleIDs

    info(s"Number of BGEN files parsed: ${bgenLoaders.length}")
    info(s"Number of variants in all BGEN files: $nVariants")
    info(s"Number of samples in BGEN files: $nSamples")
    var time = System.currentTimeMillis()

    // FIXME what about withScope and assertNotStopped()?

/*    val rdd = sc.union(bgenFiles.map{ file =>
      sc.hadoopFile(file, classOf[BgenInputFormat], classOf[LongWritable], classOf[ParsedLine[Variant]],
        nPartitions.getOrElse(sc.defaultMinPartitions))
        .map { case (lw, pl) => (pl.getKey, Annotations.empty(), pl.getGS) }
    })*/
    /*val rdd = sc.hadoopFile(file, classOf[BgenInputFormat], classOf[LongWritable], classOf[ParsedLine[Variant]],
      nPartitions.getOrElse(sc.defaultMinPartitions))
      .map { case (lw, pl) => (pl.getKey, Annotations.empty(), pl.getGS) }*/

    //require(rdd.count() == nVariants)

    //println("parsing took %.3f seconds".format((System.currentTimeMillis() - time).toDouble / 1000.0))
    VariantSampleMatrix(metadata = VariantMetadata(sampleIDs), rdd = sc.union(bgenFiles.map{ file =>
      sc.hadoopFile(file, classOf[BgenInputFormat], classOf[LongWritable], classOf[ParsedLine[Variant]],
        nPartitions.getOrElse(sc.defaultMinPartitions))
        .map { case (lw, pl) => (pl.getKey, Annotations.empty(), pl.getGS) }
    }))
  }
}