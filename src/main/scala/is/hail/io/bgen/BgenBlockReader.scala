package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

class BgenBlockReaderV12(
  job: Configuration,
  split: BgenV12InputSplit
) extends IndexedBinaryBlockReader[BgenRecordV12](job, split.fileSplit) {
  private[this] val file = split.getPath
  private[this] val bState = LoadBgen.readState(bfis)
  private[this] val indexPath = file + ".idx"

  private[this] val includeGT = job.get("includeGT").toBoolean
  private[this] val includeGP = job.get("includeGP").toBoolean
  private[this] val includeDosage = job.get("includeDosage").toBoolean
  private[this] val includeLid = job.get("includeLid").toBoolean
  private[this] val includeRsid = job.get("includeRsid").toBoolean
  private[this] var partitionFirstFileRowIdx: Long = _

  using(new IndexBTree(indexPath, job)) { btree =>
    if (split.hasFilter) {
      pos = split.keptPositions(0)
    } else {
      btree.queryArrayPositionAndFileOffset(split.getStart) match {
        case Some((arrayPosition, byteOffset)) =>
          partitionFirstFileRowIdx = arrayPosition
          pos = byteOffset
        case None =>
          pos = end
      }
    }
  }

  bfis.seek(pos)

  override def createValue(): BgenRecordV12 =
    new BgenRecordV12(bState.compressed, bState.nSamples, includeGT, includeGP, includeDosage, includeLid, includeRsid, bfis, end, partitionFirstFileRowIdx, split)

  override def next(key: LongWritable, value: BgenRecordV12): Boolean = {
    value.advance()
  }
}
