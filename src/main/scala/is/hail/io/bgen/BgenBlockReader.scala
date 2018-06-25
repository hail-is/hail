package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

class BgenBlockReaderV12(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[BgenRecordV12](job, split) {
  val file = split.getPath
  val bState = LoadBgen.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val includeGT = job.get("includeGT").toBoolean
  val includeGP = job.get("includeGP").toBoolean
  val includeDosage = job.get("includeDosage").toBoolean
  val includeLid = job.get("includeLid").toBoolean
  val includeRsid = job.get("includeRsid").toBoolean

  private[bgen] var partitionFirstFileRowIdx: Long = _

  seekToFirstBlockInSplit(split.getStart)

  def seekToFirstBlockInSplit(start: Long) {
    btree.queryArrayPositionAndFileOffset(start) match {
      case Some((arrayPosition, byteOffset)) =>
        partitionFirstFileRowIdx = arrayPosition
        pos = byteOffset
      case None =>
        pos = end
    }

    btree.close()
    bfis.seek(pos)
  }

  override def createValue(): BgenRecordV12 =
    new BgenRecordV12(bState.compressed, bState.nSamples, includeGT, includeGP, includeDosage, includeLid, includeRsid, bfis, end, partitionFirstFileRowIdx)

  override def next(key: LongWritable, value: BgenRecordV12): Boolean = {
    value.advance()
  }
}
