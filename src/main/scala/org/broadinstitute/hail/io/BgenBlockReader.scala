package org.broadinstitute.hail.io

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.{InvalidFileTypeException, FileSplit}
import org.broadinstitute.hail.variant.{Genotype, GenotypeStreamBuilder, Variant}
import org.broadinstitute.hail.Utils.hadoopOpen

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val bgenCompressed = job.getBoolean("bgenCompressed", false)
  val compressGS = job.getBoolean("compressGS", false)
  val nSamples = job.getInt("nSamples", -1)
  val version = job.getInt("version", -1)
  val indexArrayPath = job.get("idx")
  seekToFirstBlock(split.getStart)

  override def seekToFirstBlock(start: Long) {
    val path = job.get("idx")
    val position = IndexBTree.queryStart(start, hadoopOpen(path, job))
    pos = position
    println(s"seekToFirstBlock start=$start position=$position")
    bfis.seek(position)
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
//      println(s"alleles:${alleles.mkString(",")}")
//      println("nRow=%d, Lid=%s, rsid=%s, chr=%s, pos=%d, K=%d, alleles=%s".format(nRow, lid, rsid, chr, pos, nAlleles, alleles.length))
      println("nRow=%d, Lid=%s, rsid=%s, chr=%s, pos=%d, K=%d, ref=%s, alt=%s".format(nRow, lid, rsid, chr, pos, nAlleles, alleles(0),
              alleles(1)))
      // FIXME no multiallelic support (version 1.2)
      if (alleles.length > 2)
        throw new UnsupportedOperationException()

      // FIXME: using first allele as ref and second as alt
      val (variant, flip) = {
        if (alleles(0) == "R" | alleles(0) == "D" | alleles(0) == "I") {
          val munged = BgenLoader.mungeIndel(lid, alleles(0), alleles(1))
          (Variant(chr, position, munged._1, munged._2), munged._3)
        }
        // don't flip by default
        (Variant(chr, position, alleles(0), alleles(1)), false)
      }
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
      val b = new GenotypeStreamBuilder(variant, compress = bgenCompressed)

      for (i <- 0 until nSamples) {
        val pAA = bar.readShort()
        val pAB = bar.readShort()
        val pBB = bar.readShort()
        var PLs = {
          if (!flip)
            BgenLoader.phredScalePPs(pAA, pAB, pBB)
          else
            BgenLoader.phredScalePPs(pBB, pAB, pAA)
        }
        assert(PLs(0) == 0 || PLs(1) == 0 || PLs(2) == 0)
        val gtCall = BgenLoader.parseGenotype(PLs)
        PLs = if (gtCall == -1) null else PLs
        val gt = Genotype(Some(gtCall), None, None, None, Some(PLs)) // FIXME missing data for stuff
        b += gt
      }
      value.setKey(variant)
      value.setGS(b.result())
      pos = bfis.getPosition
      true
    }
  }

  def createValue(): ParsedLine[Variant] = new BgenParsedLine
}
