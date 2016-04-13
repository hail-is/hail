package org.broadinstitute.hail.io

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit
import org.broadinstitute.hail.variant.{GenotypeBuilder, GenotypeStreamBuilder, Variant}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import scala.collection.mutable

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val file = split.getPath
  val fileSize = hadoopGetFileSize(file.toString, job)
  val bState = BgenLoader.readState(bfis)
  val indexArrayPath = file + ".idx"

  val compressGS = job.getBoolean("compressGS", false)

  val ab = new mutable.ArrayBuilder.ofByte
  val plArray = new Array[Int](3)

  seekToFirstBlock(split.getStart)

  override def seekToFirstBlock(start: Long) {
    require(start >= 0 && start < fileSize)
    pos = IndexBTree.queryIndex(start, fileSize, indexArrayPath, job)
    if (pos < 0 || pos >= fileSize)
      fatal(s"incorrect seek position `$pos', the file is $fileSize bytes")
    else
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

      val ref = bfis.readLengthAndString(4)
      val alt = bfis.readLengthAndString(4)

      val variant = Variant(chr, position, ref, alt)
      val varAnnotation = Annotation(rsid, lid)

      val bytes = {
        if (bState.compressed) {
          val expansion = Array.ofDim[Byte](nRow * 6)
          val compressedBytes = bfis.readInt()
          val inflater = new Inflater
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
      var plAA = -1
      var plAB = -1
      var plBB = -1

      for (i <- 0 until bState.nSamples) {

        val pAA = bar.readShort()
        val pAB = bar.readShort()
        val pBB = bar.readShort()

        if (pAA == 32768) {
          plAA = 0
          plAB = 51
          plBB = 51
        } else if (pAB == 32768) {
          plAA = 51
          plAB = 0
          plBB = 51
        } else if (pBB == 32768) {
          plAA = 51
          plAB = 51
          plBB = 0
        } else {
          val dAA = if (pAA == 0) 51 else BgenLoader.phredConversionTable(pAA)
          val dAB = if (pAB == 0) 51 else BgenLoader.phredConversionTable(pAB)
          val dBB = if (pBB == 0) 51 else BgenLoader.phredConversionTable(pBB)

          val minValue = math.min(math.min(dAA, dAB), dBB)

          plAA = (dAA - minValue + .5).toInt
          plAB = (dAB - minValue + .5).toInt
          plBB = (dBB - minValue + .5).toInt
        }

        assert(plAA == 0 || plAB == 0 || plBB == 0)

        val gt = if (plAA == 0 && plAB == 0
          || plAA == 0 && plBB == 0
          || plAB == 0 && plBB == 0)
          -1
        else {
          if (plAA == 0)
            0
          else if (plAB == 0)
            1
          else
            2
        }

        genoBuilder.clear()
        if (gt >= 0) {
          genoBuilder.setGT(gt)
          plArray(0) = plAA
          plArray(1) = plAB
          plArray(2) = plBB
          genoBuilder.setPL(plArray)
        }
        b.write(genoBuilder)
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
