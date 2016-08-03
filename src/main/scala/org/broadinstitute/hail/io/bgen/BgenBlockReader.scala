package org.broadinstitute.hail.io.bgen

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.io._
import org.broadinstitute.hail.variant.{GenotypeBuilder, GenotypeStreamBuilder, Variant}
import org.broadinstitute.hail.io.gen.GenReport._
import org.broadinstitute.hail.io.gen.GenUtils._
import org.broadinstitute.hail.Utils._

import scala.collection.mutable

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val file = split.getPath
  val bState = BgenLoader.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val compressGS = job.getBoolean("compressGS", false)
  val tolerance = job.get("tolerance", "0.02").toDouble

  val ab = new mutable.ArrayBuilder.ofByte
  val bgenIntArray = new Array[Int](3)
  val dosageArray = new Array[Int](3)
  val dosageDivisor = 32768
  val genoBuilder = new GenotypeBuilder(2, isDosage = true)
  val b = new GenotypeStreamBuilder(2, isDosage = true, compress = compressGS)

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
      val nAlleles = 2
      val nGenotypes = 3

      val recodedChr = chr match {
        case "23" => "X"
        case "24" => "Y"
        case "25" => "X"
        case "26" => "MT"
        case x => x
      }

      val variant = Variant(recodedChr, position, ref, alt)

      value.resetWarnings()

      val bytes = {
        if (bState.compressed) {
          val expansion = Array.ofDim[Byte](nRow * 6)
          val compressedBytes = bfis.readInt()
          val inflater = new Inflater
          inflater.setInput(bfis.readBytes(compressedBytes))
          while (!inflater.finished()) {
            inflater.inflate(expansion)
          }
          expansion
        } else
          bfis.readBytes(nRow * 6)
      }

      assert(bytes.length == nRow * 6)

      b.clear()
      val bar = new ByteArrayReader(bytes)

      for (i <- 0 until bState.nSamples) {
        genoBuilder.clear()

        bgenIntArray(0) = bar.readShort()
        bgenIntArray(1) = bar.readShort()
        bgenIntArray(2) = bar.readShort()
        val (cp1, cp2, cp3) = (bgenIntArray(0), bgenIntArray(1), bgenIntArray(2))
        val intSum = bgenIntArray.sum
        val dosageSum = intSum / dosageDivisor.toDouble
        if (dosageSum == 0.0)
          value.setWarning(dosageNoCall)
        else if (1d - dosageSum > tolerance)
          value.setWarning(dosageLessThanTolerance)
        else if (dosageSum - 1d > tolerance)
          value.setWarning(dosageGreaterThanTolerance)
        else {
          dosageArray(0) = (bgenIntArray(0) / dosageSum + .5).toInt
          dosageArray(1) = (bgenIntArray(1) / dosageSum + .5).toInt
          dosageArray(2) = (bgenIntArray(2) / dosageSum + .5).toInt

          val normalizedSum = dosageArray.sum
          assert(normalizedSum > dosageDivisor - 4 && normalizedSum < dosageDivisor + 4)
          val maxIntDosage = dosageArray.max
          val gt = {
            if (maxIntDosage < 16384 && dosageArray.count(_ == maxIntDosage) != 1)
              -1
            else
              dosageArray.indexOf(maxIntDosage)
          }

          if (gt >= 0) {
            genoBuilder.setGT(gt)
          }
          genoBuilder.setPX(dosageArray)
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
