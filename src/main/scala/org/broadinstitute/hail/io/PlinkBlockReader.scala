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
//  println(s"split starts at ${split.getStart}, ends at ${split.getLength + split.getStart}")
  var variantIndex = 0
  val nSamples = job.getInt("nSamples", 0)
//  println(s"nSamples in blockReader = $nSamples")
  val compressGS = job.getBoolean("compressGS", false)
  val blockLength = ((nSamples / 4.00) + .75).toInt
//  println(s"blockLength=$blockLength")
//  println(s"expected blocks = ${(end - partitionStart) / blockLength}")
  seekToFirstBlock(split.getStart)

  def seekToFirstBlock(start: Long) {
  //  println(s"seekToFirstBlock::start = $start")
    variantIndex = ((start - 3) / blockLength).toInt
    pos = variantIndex * blockLength + 3
    if (pos < start){
      variantIndex += 1
      pos = variantIndex * blockLength + 3
    }
  //  println(s"seekToFirstBlock::variantIndex = $variantIndex")
    bfis.seek(variantIndex * blockLength + 3)


   // println(s"seekToFirstBlock::pos = $pos")
    //println(s"started first variant at $pos")
  }

  def next(key: LongWritable, value: ParsedLine[Int]): Boolean = {
    //val nullVariant = Variant("0",0,"A","T")
    //val b = new GenotypeStreamBuilder(nullVariant, compress = false)
    if (pos >= end)
      false
    else {
      val nullVariant = Variant("0",0,"A","T")
      val b = new GenotypeStreamBuilder(nullVariant, compress = compressGS)
      //println(s"index $variantIndex, pos = $pos, starts at ${bfis.getPosition}")
      bfis.readBytes(blockLength)
        .iterator
        .flatMap { i => Iterator(i & 3, (i >> 2) & 3, (i >> 4) & 3, (i >> 6) & 3) }
        .take(nSamples)
        .foreach(i => b += PlinkBlockReader.plinkOrderedSparseGt(i))
      value.setGS(b.result())
      value.setKey(variantIndex)
      variantIndex += 1
      pos += blockLength
      //      println(s"read variant $variantIndex")
      true
    }
  }

  def createValue(): PlinkParsedLine = new PlinkParsedLine
}
