package is.hail.io.bgen

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import is.hail.io.{IndexedBinaryInputFormat, KeySerializedValueRecord}
import is.hail.variant.Variant

class BgenInputFormat extends IndexedBinaryInputFormat[BgenRecord] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    BgenRecord] = {
    reporter.setStatus(split.toString)
    new BgenBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
