package is.hail.io.plink

import is.hail.annotations.Annotation
import is.hail.io.{IndexedBinaryBlockReader, KeySerializedValueRecord}
import is.hail.utils.ArrayBuilder
import is.hail.variant.{Genotype, GenotypeBuilder}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

class PlinkRecord(nSamples: Int) extends KeySerializedValueRecord[Int, IndexedSeq[Annotation]] {
  override def getValue: IndexedSeq[Annotation] = {
    require(input != null, "called getValue before serialized value was set")
    new IndexedSeq[Annotation] {
      def length: Int = nSamples
      def apply(i: Int): Annotation = {
        val x = (input(i / 4) >> ((i & 3) << 1)) & 3
        val gt = if (x == 1)
          null
        else
          PlinkBlockReader.GT_CONVERSION(x)
        Annotation(gt)
      }
    }
  }
}

object PlinkBlockReader {
  final val GT_CONVERSION = Array(2, -1, 1, 0)
}

class PlinkBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[PlinkRecord](job, split) {
  var variantIndex: Long = 0L
  val nSamples = job.getInt("nSamples", 0)
  val blockLength = (nSamples + 3) / 4

  seekToFirstBlockInSplit(split.getStart)

  override def createValue(): PlinkRecord = new PlinkRecord(nSamples)

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
