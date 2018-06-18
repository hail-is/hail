package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

abstract class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[BgenRecordV12](job, split) {
  val file = split.getPath
  val bState = LoadBgen.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val includeGT = job.get("includeGT").toBoolean
  val includeGP = job.get("includeGP").toBoolean
  val includeDosage = job.get("includeDosage").toBoolean

  seekToFirstBlockInSplit(split.getStart)

  override def createValue(): BgenRecordV12

  def seekToFirstBlockInSplit(start: Long) {
    pos = btree.queryIndex(start) match {
      case Some(x) => x
      case None => end
    }

    btree.close()
    bfis.seek(pos)
  }

  def next(key: LongWritable, value: BgenRecordV12): Boolean
}

class BgenBlockReaderV12(job: Configuration, split: FileSplit) extends BgenBlockReader(job, split) {
  override def createValue(): BgenRecordV12 =
    new BgenRecordV12(bState.compressed, bState.nSamples, includeGT, includeGP, includeDosage, bfis, end)

  override def next(key: LongWritable, value: BgenRecordV12): Boolean = {
    value.advance()
  }
}
