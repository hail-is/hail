package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io._
import is.hail.variant.Variant
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

abstract class BgenBlockReader[T <: BgenRecord](job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[T](job, split) {
  val file = split.getPath
  val bState = BgenLoader.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val tolerance = job.get("tolerance").toDouble

  seekToFirstBlockInSplit(split.getStart)

  override def createValue(): T

  def seekToFirstBlockInSplit(start: Long) {
    pos = btree.queryIndex(start) match {
      case Some(x) => x
      case None => end
    }

    btree.close()
    bfis.seek(pos)
  }

  def next(key: LongWritable, value: T): Boolean
}

class BgenBlockReaderV11(job: Configuration, split: FileSplit) extends BgenBlockReader[BgenRecordV11](job, split) {
  override def createValue(): BgenRecordV11 = new BgenRecordV11(bState.compressed, bState.nSamples, tolerance)

  override def next(key: LongWritable, value: BgenRecordV11): Boolean = {
    if (pos >= end)
      false
    else {
      val nRow = bfis.readInt()

      // we silently assumed this in previous iterations of the code.  Now explicitly assume.
      assert(nRow == bState.nSamples, "row nSamples is not equal to header nSamples")

      val lid = bfis.readLengthAndString(2)
      val rsid = bfis.readLengthAndString(2)
      val chr = bfis.readLengthAndString(2)
      val position = bfis.readInt()

      val ref = bfis.readLengthAndString(4)
      val alt = bfis.readLengthAndString(4)

      val recodedChr = chr match {
        case "23" => "X"
        case "24" => "Y"
        case "25" => "X"
        case "26" => "MT"
        case x => x
      }

      val variant = Variant(recodedChr, position, ref, alt)

      val bytesInput = if (bState.compressed) {
        val compressedBytes = bfis.readInt()
        bfis.readBytes(compressedBytes)
      } else
        bfis.readBytes(nRow * 6)

      value.setKey(variant)
      value.setAnnotation(Annotation(rsid, lid))
      value.setSerializedValue(bytesInput)

      pos = bfis.getPosition
      true
    }
  }
}

class BgenBlockReaderV12(job: Configuration, split: FileSplit) extends BgenBlockReader[BgenRecordV12](job, split) {
  override def createValue(): BgenRecordV12 = new BgenRecordV12(bState.compressed, bState.nSamples, tolerance)

  override def next(key: LongWritable, value: BgenRecordV12): Boolean = {
    if (pos >= end)
      false
    else {
      val lid = bfis.readLengthAndString(2)
      val rsid = bfis.readLengthAndString(2)
      val chr = bfis.readLengthAndString(2)
      val position = bfis.readInt()

      val nAlleles = bfis.readShort()
      assert(nAlleles >= 2, s"Number of alleles must be greater than or equal to 2. Found $nAlleles alleles for variant '$lid'")
      val nAltAlleles = nAlleles - 1

      val ref = bfis.readLengthAndString(4)
      val altAlleles = new Array[String](nAltAlleles)

      var altIndex = 0
      while (altIndex < nAltAlleles) {
        altAlleles(altIndex) = bfis.readLengthAndString(4)
        altIndex += 1
      }

      val recodedChr = chr match {
        case "23" => "X"
        case "24" => "Y"
        case "25" => "X"
        case "26" => "MT"
        case x => x
      }

      val variant = Variant(recodedChr, position, ref, altAlleles)

      val dataSize = bfis.readInt()

      val (uncompressedSize, bytesInput) =
        if (bState.compressed)
          (bfis.readInt(), bfis.readBytes(dataSize - 4))
        else
          (dataSize, bfis.readBytes(dataSize))

      value.setKey(variant)
      value.setAnnotation(Annotation(rsid, lid))
      value.setSerializedValue(bytesInput)
      value.setExpectedDataSize(uncompressedSize)
      value.setExpectedNumAlleles(nAlleles)

      pos = bfis.getPosition
      true
    }
  }
}
