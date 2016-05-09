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
      val nAlleles = 2 //FIXME: for v1.2 the number of alleles is variable
      val nGenotypes = triangle(nAlleles)

      val variant = Variant(chr, position, ref, alt)

      value.resetGenotypeFlags()

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

        val origDosages = (0 until nGenotypes).map{case i => bar.readShort() / 32768.0}.toArray
        val sumDosages = origDosages.sum

        if (sumDosages == 0.0)
          value.setGenotypeFlags(dosageNoCall)
        else if (math.abs(sumDosages - 1.0) > tolerance)
          value.setGenotypeFlags(dosageLessThanTolerance)
        else {
          val normIntDosages = normalizePPs(origDosages).map(convertProbToInt)
          val sumIntDosages = normIntDosages.sum
          assert(sumIntDosages >= 32768 - nGenotypes && sumIntDosages <= 32768 + nGenotypes)

          val maxIntDosage = normIntDosages.max
          val gt = {
            if (maxIntDosage < 16384 && normIntDosages.count(_ == maxIntDosage) != 1) //first comparison for speed to not evaluate count if prob > 0.5
              -1
            else
              normIntDosages.indexOf(maxIntDosage)
          }

          if (gt >= 0) {
            genoBuilder.setGT(gt)
          }

          genoBuilder.setDosage(normIntDosages)
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
