package org.broadinstitute.hail.io

import java.util.zip.Inflater

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.{InvalidFileTypeException, FileSplit}
import org.broadinstitute.hail.variant.{Genotype, GenotypeStreamBuilder, Variant}
import org.broadinstitute.hail.Utils._

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Variant](job, split) {
  val file = split.getPath
  val indexArrayPath = file + ".idx"
  val compressGS = job.getBoolean("compressGS", false)

  var nSamples = 0
  var nVariants = 0
  var bgenCompressed = false
  var version = -1

  readFileParameters()
  seekToFirstBlock(split.getStart)

  def readFileParameters() {
    bfis.seek(0)
    val offset = bfis.readInt()

    bfis.seek(8)
    nVariants = bfis.readInt()
    nSamples = bfis.readInt()

    bfis.seek(offset)
    val flags = bfis.readInt()
    bgenCompressed = (flags & 1) != 0 // either 0 or 1 based on the first bit
    version = flags >> 2 & 0xf
  }

  override def seekToFirstBlock(start: Long) {
    pos = IndexBTree.queryStart(start, hadoopOpen(indexArrayPath, job))
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

/*      println("nRow=%d, Lid=%s, rsid=%s, chr=%s, pos=%d, K=%d, ref=%s, alt=%s".format(nRow, lid, rsid, chr, pos, nAlleles, alleles(0),
              alleles(1)))*/

      // FIXME no multiallelic support (version 1.2)
      if (alleles.length > 2)
        throw new UnsupportedOperationException()

      // FIXME: using first allele as ref and second as alt
      val variant = Variant(chr, position, alleles(0), alleles(1))

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

        var PLs = BgenLoader.phredScalePPs(pAA, pAB, pBB)

        assert(PLs(0) == 0 || PLs(1) == 0 || PLs(2) == 0)
        val gtCall = BgenLoader.parseGenotype(PLs)
        PLs = if (gtCall == -1) null else PLs

        val gt = Genotype(Option(gtCall), None, None, None, Option(PLs))
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
