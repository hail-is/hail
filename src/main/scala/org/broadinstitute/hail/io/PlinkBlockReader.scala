package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.{RecordReader, FileSplit}
import org.broadinstitute.hail.variant.{GenotypeStreamBuilder, Genotype, Variant}

object PlinkBlockReader {
  val sparseGt = Array(Genotype(-1, (0, 0), 0, (0, 0, 0)),
    Genotype(0, (0, 0), 0, (0, 0, 0)),
    Genotype(1, (0, 0), 0, (0, 0, 0)),
    Genotype(2, (0, 0), 0, (0, 0, 0)))
}

class PlinkBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[Int](job, split) {

  var variantIndex = 0
  val nSamples = job.getInt("nSamples", 0)
  val compressGS = job.getBoolean("compressGS", false)
  val blockLength = ((nSamples / 4.00) + .75).toInt

  def getFirstBlock(start: Long): Long = {
    variantIndex = ((start - 3) / blockLength).toInt
    variantIndex * blockLength + 3
  }

  def plinkToHail(call: Int): Int = {
    if (call == 0)
      call
    else if (call == 1)
      -1
    else
      call - 1
  }

  def next(key: LongWritable, value: ParsedLine[Int]): Boolean = {
    val b = new GenotypeStreamBuilder(null, compress = false)
    if (pos >= end)
      false
    else {
      val b = new GenotypeStreamBuilder(null, compress = compressGS)
      bfis.readBytes(blockLength)
        .iterator
        .flatMap { i => Iterator(i & 3, (i >> 2) & 3, (i >> 4) & 3, (i >> 6) & 3) }
        .map(plinkToHail)
        .take(nSamples)
        .foreach(i => b += PlinkBlockReader.sparseGt(i + 1))
      value.setGS(b.result())
      value.setKey(variantIndex)
      variantIndex += 1
      true
    }
  }

  def createValue(): PlinkParsedLine = new PlinkParsedLine
}
