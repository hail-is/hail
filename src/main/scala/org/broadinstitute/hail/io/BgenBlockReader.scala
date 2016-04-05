package org.broadinstitute.hail.io

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit
import org.broadinstitute.hail.variant.{Genotype, GenotypeBuilder, GenotypeStreamBuilder, Variant}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.annotations._
import scala.collection.mutable

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val file = split.getPath
  val fileSize = hadoopGetFileSize(file.toString, job)
  val indexArrayPath = file + ".idx"
  val compressGS = job.getBoolean("compressGS", false)

  var nSamples = 0
  var nVariants = 0
  var bgenCompressed = false
  var version = -1
  val ab = new mutable.ArrayBuilder.ofByte
  println(ab.getClass.getName)

  readFileParameters()
  seekToFirstBlock(split.getStart)

  def readFileParameters() {
    bfis.seek(0)
    val offset = bfis.readInt()
    val headerLength = bfis.readInt()
    nVariants = bfis.readInt()
    nSamples = bfis.readInt()
    val magicNumber = bfis.readString(4) //readers ignore these bytes

    val headerInfo = {
      if (headerLength > 20)
        bfis.readString(headerLength.toInt - 20)
      else
        ""
    }

    val flags = bfis.readInt()

    bgenCompressed = (flags & 1) != 0 // either 0 or 1 based on the first bit
    version = (flags >> 2) & 0xf
  }

  override def seekToFirstBlock(start: Long) {
    require(start >= 0 && start < fileSize)
    pos = IndexBTree.queryIndex(start, fileSize, indexArrayPath, job)
    if (pos < 0 || pos > fileSize)
      fatal(s"incorrect seek position: pos=$pos start=$start fileSize=$fileSize")
    if (pos >= 0 && pos < fileSize)
      bfis.seek(pos)
  }

  def next(key: LongWritable, value: ParsedLine[Variant]): Boolean = {
    if (pos >= end)
      false
    else {
      val nRow = bfis.readInt()
      val lid = bfis.readLengthAndString(2)
      val rsid = bfis.readLengthAndString(2)
      val chr = bfis.readLengthAndString(2)
      val position = bfis.readInt()


      val nAlleles = if (version == 1) 2 else bfis.readShort()
      val alleles = Array.ofDim[String](nAlleles)

      for (i <- 0 until nAlleles) {
        alleles(i) = bfis.readLengthAndString(4)
      }


      // FIXME no multiallelic support (version 1.2)
      if (alleles.length > 2)
        throw new UnsupportedOperationException()

      // FIXME: using first allele as ref and second as alt
      val variant = Variant(chr, position, alleles(0), alleles(1))
      val varAnnotation = Annotation(rsid, lid) //order must match TStruct

      val bytes = {
        if (bgenCompressed) {
          val expansion = Array.ofDim[Byte](nRow * 6)
          val inflater = new Inflater()
          val compressedBytes = bfis.readInt()
          inflater.setInput(bfis.readBytes(compressedBytes))
          var decompressed = 0
          while (!inflater.finished()) {
            inflater.inflate(expansion)
          }
          expansion
        }
        else
          bfis.readBytes(nRow * 6)
      }

      assert(bytes.length == nRow * 6)

      val bar = new ByteArrayReader(bytes)
      val b = new GenotypeStreamBuilder(variant, ab, compress = compressGS)

      val genoBuilder = new GenotypeBuilder(variant)
      val plArray = Array(-1, -1, -1)

      for (i <- 0 until nSamples) {

        val pAA = bar.readShort()
        val pAB = bar.readShort()
        val pBB = bar.readShort()
        if (lid == "SNPID_99")
          if (i == 345)
            println(pAA, pAB, pBB)

        if (pAA == 32768) {
          plArray(0) = 0
          plArray(1) = 51
          plArray(2) = 51
        } else if (pAB == 32768) {
          plArray(0) = 51
          plArray(1) = 0
          plArray(2) = 51
        } else if (pBB == 32768) {
          plArray(0) = 51
          plArray(1) = 51
          plArray(2) = 0
        } else {
          val dAA = if (pAA == 0) 51 else BgenLoader.phredConversionTable(pAA)
          val dAB = if (pAB == 0) 51 else BgenLoader.phredConversionTable(pAB)
          val dBB = if (pBB == 0) 51 else BgenLoader.phredConversionTable(pBB)

          val minValue = math.min(math.min(dAA, dAB), dBB)

          plArray(0) = (dAA - minValue + .5).toInt
          plArray(1) = (dAB - minValue + .5).toInt
          plArray(2) = (dBB - minValue + .5).toInt
          //            if (lid == "SNPID_99")
          //              if (i == 345)
          //                println(Array((dAA - minValue + .5).toInt,
          //                  (dAB - minValue + .5).toInt,
          //                  (dBB - minValue + .5).toInt).mkString(", "))
        }

        assert(plArray(0) == 0 || plArray(1) == 0 || plArray(2) == 0)

        val gt = if (plArray(0) == 0 && plArray(1) == 0
          || plArray(0) == 0 && plArray(2) == 0
          || plArray(1) == 0 && plArray(2) == 0)
          -1
        else {
          if (plArray(0) == 0)
            0
          else if (plArray(1) == 0)
            1
          else
            2
        }

        genoBuilder.clear()
        if (gt >= 0) {
          genoBuilder.setGT(gt)
          genoBuilder.setPL(plArray)
        }
        b.write(genoBuilder)
        //        val genotype = Genotype(Option(gt), None, None, None, if (gt < 0) None else Some(PLs))
        //        b += genotype
      }

      value.setKey(variant)
      value.setAnnotation(varAnnotation)
      value.setGS(b.result())
      pos = bfis.getPosition
      true
    }
  }

  def createValue(): ParsedLine[Variant] = new BgenParsedLine
}
