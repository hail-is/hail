package org.broadinstitute.hail.io.bgen

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.io._
import org.broadinstitute.hail.variant.{GenotypeBuilder, GenotypeStreamBuilder, Variant}

import scala.collection.mutable

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val file = split.getPath
  //val fileSize = hadoopGetFileSize(file.toString, job)
  val bState = BgenLoader.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val compressGS = job.getBoolean("compressGS", false)

  val ab = new mutable.ArrayBuilder.ofByte
  val plArray = new Array[Int](3)

  seekToFirstBlockInSplit(split.getStart)

  def seekToFirstBlockInSplit(start: Long) {
    pos = btree.queryIndex(start) match {
      case Some(x) => x
      case None => end
    }

    btree.close()
    bfis.seek(pos)
  }

  def next(key: LongWritable, value: VariantRecord[Variant]): Boolean = {
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
        } else
          bfis.readBytes(nRow * 6)
      }

      assert(bytes.length == nRow * 6)

      val bar = new ByteArrayReader(bytes)
      val b = new GenotypeStreamBuilder(variant, compress = compressGS)

      val genoBuilder = new GenotypeBuilder(variant)

      for (i <- 0 until bState.nSamples) {

        val pAA = bar.readShort()
        val pAB = bar.readShort()
        val pBB = bar.readShort()

        val dAA = BgenLoader.phredConversionTable(pAA)
        val dAB = BgenLoader.phredConversionTable(pAB)
        val dBB = BgenLoader.phredConversionTable(pBB)

        val minValue = math.min(math.min(dAA, dAB), dBB)

        val plAA = (dAA - minValue + .5).toInt
        val plAB = (dAB - minValue + .5).toInt
        val plBB = (dBB - minValue + .5).toInt

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
}
