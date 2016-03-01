package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.{RecordReader, FileSplit}
import org.broadinstitute.hail.variant.{GenotypeStreamBuilder, Genotype, Variant}

object PlinkBlockReader {
  val plinkOrderedSparseGt = Array(
    Genotype(2),
    Genotype(-1),
    Genotype(1),
    Genotype(0))
}

class PlinkBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Int](job, split) {
  var variantIndex = 0
  val nSamples = job.getInt("nSamples", 0)
  val compressGS = job.getBoolean("compressGS", false)
  val blockLength = ((nSamples / 4.00) + .75).toInt

  seekToFirstBlock(split.getStart)

  def seekToFirstBlock(start: Long) {
    variantIndex = ((start - 3) / blockLength).toInt

    pos = variantIndex * blockLength + 3
    if (pos < start){
      variantIndex += 1
      pos = variantIndex * blockLength + 3
    }

    bfis.seek(variantIndex * blockLength + 3)
  }

  def next(key: LongWritable, value: ParsedLine[Int]): Boolean = {
    if (pos >= end)
      false
    else {
      val nullVariant = Variant("0",0,"A","T")
      val b = new GenotypeStreamBuilder(nullVariant, compress = compressGS)

      bfis.readBytes(blockLength)
        .iterator
        .flatMap { i => Iterator(i & 3, (i >> 2) & 3, (i >> 4) & 3, (i >> 6) & 3) }
        .take(nSamples)
        .foreach(i => b += PlinkBlockReader.plinkOrderedSparseGt(i))

      value.setGS(b.result())
      value.setKey(variantIndex)
      variantIndex += 1
      pos += blockLength

      true
    }
  }

  def createValue(): PlinkParsedLine = {
    new PlinkParsedLine
  }
}
