package org.broadinstitute.hail.io.bgen

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.io._
import org.broadinstitute.hail.variant.{GenotypeBuilder, GenotypeStreamBuilder, Variant}
import org.broadinstitute.hail.io.gen.{GenReport, GenUtils}

import scala.collection.mutable

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val file = split.getPath
  val bState = BgenLoader.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val compressGS = job.getBoolean("compressGS", false)
  val tolerance = job.get("tolerance", "0.02").toDouble

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
        genoBuilder.clear()

        genoBuilder.setDosageFlag()

        val pAA = bar.readShort()
        val pAB = bar.readShort()
        val pBB = bar.readShort()

        val origDosages = Array(pAA, pAB, pBB).map{_ / 32768.0}
        val sumDosages = origDosages.sum

        if (sumDosages == 0.0)
          value.setGenotypeFlag(GenReport.dosageNoCall)
        else if (math.abs(origDosages.sum - 1.0) > tolerance)
          value.setGenotypeFlag(GenReport.dosageLessThanTolerance)
        else {
          val normProbs = GenUtils.normalizePPs(origDosages)

          val dosageAA = GenUtils.convertProbsToInt(normProbs(0))
          val dosageAB = GenUtils.convertProbsToInt(normProbs(1))
          val dosageBB = GenUtils.convertProbsToInt(normProbs(2))

          val sumDosage = dosageAA + dosageAB + dosageBB

          assert(sumDosage >= 32765 && sumDosage <= 32771)

          val gt = {
            if (dosageAA > dosageAB && dosageAA > dosageBB)
              0
            else if (dosageAB > dosageAA && dosageAB > dosageBB)
              1
            else if (dosageBB > dosageAA && dosageBB > dosageAB)
              2
            else
              -1
          }

          if (gt >= 0) {
            genoBuilder.setGT(gt)
          }

          genoBuilder.setDosage(Array(dosageAA, dosageAB, dosageBB))
        }

        b.write(genoBuilder)

      }

      val varAnnotation = Annotation(rsid, lid)

      value.setKey(variant)
      value.setAnnotation(varAnnotation)
      value.setGS(b.result())
      pos = bfis.getPosition
      true
    }
  }
}
