package is.hail.io.bgen

import is.hail.io.IndexedBinaryInputFormat
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

class BgenInputFormat extends IndexedBinaryInputFormat[BgenRecord] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    BgenRecord] = {
    reporter.setStatus(split.toString)
    new BgenBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
