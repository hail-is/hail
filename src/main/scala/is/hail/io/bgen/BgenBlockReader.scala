package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

abstract class BgenBlockReader[T <: BgenRecord](job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[T](job, split) {
  val file = split.getPath
  val bState = LoadBgen.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val includeGT = job.get("includeGT").toBoolean
  val includeGP = job.get("includeGP").toBoolean
  val includeDosage = job.get("includeDosage").toBoolean
  val includeLid = job.get("includeLid").toBoolean
  val includeRsid = job.get("includeRsid").toBoolean

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

class BgenBlockReaderV12(
  job: Configuration,
  split: BgenV12InputSplit
) extends BgenBlockReader[BgenRecordV12](job, split.fileSplit) {
  override def createValue(): BgenRecordV12 = new BgenRecordV12(
    bState.compressed,
    bState.nSamples,
    includeGT,
    includeGP,
    includeDosage,
    bfis)

  private[this] var i = 0

  override def next(key: LongWritable, value: BgenRecordV12): Boolean = {
    if (split.keptPositions != null) {
      if (i >= split.keptPositions.length)
        pos = end
      else
        pos = split.keptPositions(i)
      bfis.seek(pos)
    }
    i += 1

    if (bfis.getPosition >= end)
      false
    else {
      val start = bfis.getPosition
      val lid = if (includeLid)
        bfis.readLengthAndString(2)
      else {
        bfis.readLengthAndSkipString(2)
        null
      }
      val rsid = if (includeRsid)
        bfis.readLengthAndString(2)
      else {
        bfis.readLengthAndSkipString(2)
        null
      }
      val chr = bfis.readLengthAndString(2)
      val position = bfis.readInt()

      val nAlleles = bfis.readShort()
      if (!(nAlleles >= 2))
        fatal(s"Number of alleles must be greater than or equal to 2. Found $nAlleles alleles for variant $chr:$pos ($lid, $rsid) $i")
      val alleles = new Array[String](nAlleles)

      val ref = bfis.readLengthAndString(4)
      alleles(0) = ref

      var aIdx = 1
      while (aIdx < nAlleles) {
        alleles(aIdx) = bfis.readLengthAndString(4)
        aIdx += 1
      }

      val recodedChr = chr match {
        case "23" => "X"
        case "24" => "Y"
        case "25" => "X"
        case "26" => "MT"
        case x => x
      }

      val variantInfo = (recodedChr, position, alleles)

      val dataSize = bfis.readInt()

      val uncompressedSize =
        if (bState.compressed) bfis.readInt() else dataSize

      value.setKey(variantInfo)
      if (includeLid || includeRsid)
        value.setAnnotation(Annotation(rsid, lid))
      value.dataSize = if (bState.compressed) dataSize - 4 else dataSize
      value.setExpectedDataSize(uncompressedSize)
      value.setExpectedNumAlleles(nAlleles)

      true
    }
  }
}
