package org.broadinstitute.hail.io

import java.io._
import java.nio.file._
import java.util.zip.Inflater
import org.apache.hadoop.io.LongWritable

import scala.io.Source
import org.broadinstitute.hail.variant._
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer

class BgenLoader(file: String, sc: SparkContext) {
  private var compression: Boolean = false
  private var version: Int = 0
  private var hasSampleIdBlock: Boolean = false
  private var sampleIDs: Option[Array[String]] = None
  private val reader = new BinaryFileReader(new RandomAccessFile(file, "r"))
  var nSamples: Int = 0
  var nVariants: Int = 0

  def getNSamples: Int = nSamples

  def getNVariants: Int = nVariants

  private def getNextBlockPosition(position: Long): Long = {
    // First seek to the proper location
    reader.seek(position)

    // Read through the variant ID block
    var nRow = nSamples
    if (version == 1)
      nRow = reader.readInt()
    reader.readLengthAndString(2) // Lid
    reader.readLengthAndString(2) // rsid
    reader.readLengthAndString(2) // chr
    reader.readInt() // pos
    val nAlleles = if (version == 1) 2 else reader.readShort()
    for (i <- 0 until nAlleles)
      reader.readLengthAndString(4) // read an allele

    // Read the size of the genotype probability block
    if (version == 1 && !compression)
      reader.getPosition + 6 * nRow
    else if (version == 1 && compression) {
      // The following works for 1.1 compressed
      val nextOffset = reader.readInt()
      nextOffset + reader.getPosition
    }
    else
      throw new UnsupportedOperationException()
  }

  private def getParseFunction = {
    val localCompression = compression
    val localNSamples = nSamples
    val localVersion = version

    (bb: ByteBlock) => {
      val bbis = new ByteArrayReader(bb)
      val nRow = bbis.readInt()
      val lid = bbis.readLengthAndString(2)
      val rsid = bbis.readLengthAndString(2)
      val chr = bbis.readLengthAndString(2)
      val pos = bbis.readInt()
      val nAlleles = if (localVersion == 1) 2 else bbis.readShort()
      val alleles = Array.ofDim[String](nAlleles)
      for (i <- 0 until nAlleles) {
        alleles(i) = bbis.readLengthAndString(4)
      }
//      println("nRow=%d, Lid=%s, rsid=%s, chr=%s, pos=%d, K=%d, ref=%s, alt=%s".format(nRow, lid, rsid, chr, pos, nAlleles, alleles(0),
//        alleles(1)))
      // FIXME no multiallelic support (version 1.2)
      if (alleles.length > 2)
        throw new UnsupportedOperationException()

      // FIXME: using first allele as ref and second as alt
      val (variant, flip) = {
        if (alleles(0) == "R" | alleles(0) == "D" | alleles(0) == "I") {
          val munged = BgenLoader.mungeIndel(lid, alleles(0), alleles(1))
          (Variant(chr, pos, munged._1, munged._2), munged._3)
        }
        // don't flip by default
        (Variant(chr, pos, alleles(0), alleles(1)), false)
      }
      val bytes = {
        if (localCompression) {
          val expansion = Array.ofDim[Byte](nRow * 6)
          val inflater = new Inflater()
          val compressedBytes = bbis.readInt()
          inflater.setInput(bbis.readBytes(compressedBytes))
          var decompressed = 0
          while (!inflater.finished()) {
            inflater.inflate(expansion)
          }
          expansion
        }
        else
          bbis.readBytes(nRow * 6)
      }

      assert(bytes.length == nRow * 6)

      val bar = new ByteArrayReader(bytes)
      val b = new GenotypeStreamBuilder(variant)

      for (i <- 0 until localNSamples) {
        val pAA = bar.readShort() / 32768.0
        val pAB = bar.readShort() / 32768.0
        val pBB = bar.readShort() / 32768.0
        var PLs = {
          if (!flip)
            BgenLoader.phredScalePPs(pAA, pAB, pBB)
          else
            BgenLoader.phredScalePPs(pBB, pAB, pAA)
        }
        assert(PLs._1 == 0 || PLs._2 == 0 || PLs._3 == 0)
        val gtCall = BgenLoader.parseGenotype(PLs)
        PLs = if (gtCall == -1) null else PLs
        val gt = Genotype(gtCall, (0, 0), 0, PLs) // FIXME missing data for stuff
        b.+=((0, gt))
      }
      (variant, b.result())
    }
  }


  def parseHeaderAndIndex(): (Option[Array[String]]) = {
    // read the length of all header stuff
    reader.seek(0)
    val allInfoLength = reader.readInt()
//    println("allInfoLen is " + allInfoLength)

    // read the header block
    val headerLength = reader.readInt()
//    println("header len is " + headerLength)
    nVariants = reader.readInt()
    println("nVariants is " + nVariants)
    nSamples = reader.readInt()
    println("nSamples is " + nSamples)
    val magicNumber = reader.readString(4)
    //    println("magic number is " + magicNumber)
    //    assert(magicNumber == "bgen")  // four zero-bytes are what, exactly? "\0\0\0\0"?

    val headerInfo = {
      if (headerLength > 20)
        reader.readString(headerLength.toInt - 20)
      else
        ""
    }

    // Parse flags
    val flags = reader.readInt()
    compression = (flags & 1) != 0 // either 0 or 1 based on the first bit
    version = flags >> 2 & 0xf
    hasSampleIdBlock = (flags >> 30 & 1) != 0
    require(version == 0 || version == 1 || version == 2)

    // version 1.2 is currently unsupported
    if (version == 2)
      throw new NotImplementedError("Support for bgen version 1.2 has not been written")

    println("flags stuff: compression=" + compression + ", version=" + version + ", hasSampleID=" + hasSampleIdBlock)

    // Read sample ID block if it has one
    val sampleIDs = hasSampleIdBlock match {
      case true =>
        val sampleIdSize = reader.readInt()
        println("sampleIdSize is " + sampleIdSize)
        val nSamplesConfirmation = reader.readInt()
        println("nSamplesConfirm=" + nSamplesConfirmation)
        assert(nSamplesConfirmation == nSamples)

        val sampleIdArr = new Array[String](nSamples)
        for (i <- 0 until nSamples) {
          sampleIdArr(i) = reader.readLengthAndString(2)
        }
        Some(sampleIdArr)
      case false => None
    }
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

    ioUtils.writeIndexArray(dataBlockStarts, file + ".idx")

    sampleIDs
  }
}

object BgenLoader {

  def mungeIndel(id: String, a1: String, a2: String): (String, String, Boolean) = {
    val alleles = id.split(":").last.split("_")
    val alt: String = {
      if (alleles.length == 1)
        alleles(0)
      else if (alleles(0).length > alleles(1).length)
        alleles(0)
      else
        alleles(1)
    }
    val ref: String = alt(0).toString

    if (a1 == "R")
      (ref, alt, false)
    else
      (ref, alt, true)
  }

  def parseGenotype(pls: (Int, Int, Int)): Int = {
    if (pls._1 == 0 && pls._2 == 0
      || pls._1 == 0 && pls._3 == 0
      || pls._2 == 0 && pls._3 == 0)
      -1
    else {
      if (pls._1 == 0)
        0
      else if (pls._2 == 0)
        1
      else if (pls._3 == 0)
        2
      else
        -1
    }
  }

  def phredScale(x: Double): Int = ((-10 * math.log10(math.max(x, math.pow(10, -10)))) + .5).toInt

  def phredScalePPs(probAA: Double, probAB: Double, probBB: Double): (Int, Int, Int) = {
    val maxValue = math.max(math.max(probAA, probAB), probBB)
    (phredScale(probAA / maxValue), phredScale(probAB / maxValue), phredScale(probBB / maxValue))
  }

  def apply(file: String, sc: SparkContext, vsmType: String = "sparky"): VariantDataset = {
    require(vsmType == "sparky" | vsmType == "tuple")
    val bl = new BgenLoader(file, sc)

    val sampleIDs = bl.parseHeaderAndIndex()
    val nSamples = bl.getNSamples
    val nVariants = bl.getNVariants

    val ids = sampleIDs match {
      case Some(arr) => arr
      case None => (0 until nSamples).map(_.toString).toArray
    }

    val returned = ioUtils.readIndexArray(file + ".idx")

    var time = System.currentTimeMillis()

    // FIXME what about withScope and assertNotStopped()?
    sc.hadoopConfiguration.set("idx", file + ".idx")
    val parseFunction = bl.getParseFunction
    val rdd = sc.hadoopFile(file, classOf[BgenInputFormat], classOf[RichLongWritable], classOf[ByteBlock],
      sc.defaultMinPartitions).map(info => parseFunction(info._2))
    rdd.count()
    println("parsing took %.3f seconds".format((System.currentTimeMillis() - time).toDouble / 1000.0))
    VariantSampleMatrix(vsmType, VariantMetadata(null, ids), rdd)
  }
}