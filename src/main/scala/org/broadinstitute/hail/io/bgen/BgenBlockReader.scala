package org.broadinstitute.hail.io.bgen

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.io._
import org.broadinstitute.hail.variant.{Genotype, GenotypeBuilder, GenotypeStreamBuilder, Variant}
import org.broadinstitute.hail.io.gen.GenReport._

import scala.collection.mutable

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val file = split.getPath
  val bState = BgenLoader.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val compressGS = job.getBoolean("compressGS", false)
  val tolerance = job.get("tolerance", "0.02").toDouble

  val ab = new mutable.ArrayBuilder.ofByte
  val dosageDivisor = 32768
  val gb = new GenotypeBuilder(2, isDosage = true)
  val gsb = new GenotypeStreamBuilder(2, isDosage = true, compress = compressGS)

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

      gsb.clear()
      val bar = new ByteArrayReader(bytes)

      for (i <- 0 until bState.nSamples) {
        gb.clear()

        val d0 = bar.readShort()
        val d1 = bar.readShort()
        val d2 = bar.readShort()
        val dosageSum = (d0 + d1 + d2) / dosageDivisor.toDouble
        if (dosageSum == 0.0)
          value.setWarning(dosageNoCall)
        else if (1d - dosageSum > tolerance)
          value.setWarning(dosageLessThanTolerance)
        else if (dosageSum - 1d > tolerance)
          value.setWarning(dosageGreaterThanTolerance)
        else {
          val px = Genotype.weightsToLinear(d0, d1, d2)
          val gt = Genotype.gtFromLinear(px)
          gt.foreach(gt => gb.setGT(gt))
          gb.setPX(px)
        }

        gsb.write(gb)
      }

      val varAnnotation = Annotation(rsid, lid)

      value.setKey(variant)
      value.setAnnotation(varAnnotation)
      value.setGS(gsb.result())
      pos = bfis.getPosition
      true
    }
  }
}
