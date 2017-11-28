package is.hail.io.plink

import is.hail.annotations.RegionValueBuilder
import is.hail.io.{IndexedBinaryBlockReader, KeySerializedValueRecord}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

import scala.annotation.switch

class PlinkRecord(nSamples: Int, a2Reference: Boolean) extends KeySerializedValueRecord[Int] {
  override def getValue(rvb: RegionValueBuilder) {
    require(input != null, "called getValue before serialized value was set")

    val c = if (a2Reference) 0 else 3
    rvb.startArray(nSamples)
    var i = 0
    while (i < nSamples) {
      rvb.startStruct() // g
      val x = (input(i >> 2) >> ((i & 3) << 1)) & 3
      (x: @switch @unchecked) match {
        case 0 => rvb.addInt(if (a2Reference) 2 else 0)
        case 1 => rvb.setMissing()
        case 2 => rvb.addInt(1)
        case 3 => rvb.addInt(if (a2Reference) 0 else 2)
      }
      rvb.endStruct() // g
      i += 1
    }
    rvb.endArray()
  }
}

class PlinkBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[PlinkRecord](job, split) {
  var variantIndex: Long = 0L
  val nSamples = job.getInt("nSamples", 0)
  val a2Reference = job.getBoolean("a2Reference", true)
  val blockLength = (nSamples + 3) / 4

  seekToFirstBlockInSplit(split.getStart)

  override def createValue(): PlinkRecord = new PlinkRecord(nSamples, a2Reference)

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
