package org.broadinstitute.hail.io.plink

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit
import org.broadinstitute.hail.io.{KeySerializedValueRecord, IndexedBinaryBlockReader}
import org.broadinstitute.hail.variant.{Genotype, GenotypeBuilder, GenotypeStreamBuilder}

class PlinkRecord(nSamples: Int, gb: GenotypeBuilder, gsb: GenotypeStreamBuilder) extends KeySerializedValueRecord[Int, Iterable[Genotype]] {
  override def getValue: Iterable[Genotype] = {
    require(input != null, "called getValue before serialized value was set")

    gsb.clear()

    input
      .iterator
      .flatMap { i => Iterator(i & 3, (i >> 2) & 3, (i >> 4) & 3, (i >> 6) & 3) }
      .take(nSamples)
      .map(PlinkBlockReader.GT_CONVERSION)
      .foreach { i =>
        gb.clear()
        if (i >= 0)
          gb.setGT(i)
        gsb.write(gb)
      }
    gsb.result()
  }
}


object PlinkBlockReader {
  final val GT_CONVERSION = Array(2, -1, 1, 0)
}

class PlinkBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[PlinkRecord](job, split) {
  var variantIndex: Long = 0L
  val nSamples = job.getInt("nSamples", 0)
  val compressGS = job.getBoolean("compressGS", false)
  val blockLength = (nSamples + 3) / 4

  val gb = new GenotypeBuilder(2, isDosage = false)
  val gsb = new GenotypeStreamBuilder(2, isDosage = false, compress = compressGS)

  seekToFirstBlockInSplit(split.getStart)

  override def createValue(): PlinkRecord = new PlinkRecord(nSamples, gb, gsb)

  def seekToFirstBlockInSplit(start: Long) {
    variantIndex = math.max(0, (start - 3 + blockLength - 1) / blockLength)
    pos = variantIndex * blockLength + 3

    if (pos < start) {
      variantIndex += 1
      pos = variantIndex * blockLength + 3
    }

    assert(pos >= 0, s"negative seek position $pos from variant index $variantIndex and block length $blockLength")

    bfis.seek(pos)
  }

  def next(key: LongWritable, value: PlinkRecord): Boolean = {
    if (pos >= end)
      false
    else {
      value.setSerializedValue(bfis.readBytes(blockLength))

      assert(variantIndex >= 0 && variantIndex <= Integer.MAX_VALUE)
      value.setKey(variantIndex.toInt)
      variantIndex += 1
      pos += blockLength

      true
    }
  }
}
