package is.hail.io.plink

import is.hail.io.IndexedBinaryInputFormat
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

class PlinkInputFormat extends IndexedBinaryInputFormat[PlinkRecord] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    PlinkRecord] = {
    reporter.setStatus(split.toString)
    new PlinkBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
