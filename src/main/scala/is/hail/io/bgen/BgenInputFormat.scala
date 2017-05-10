package is.hail.io.bgen

import is.hail.io.IndexedBinaryInputFormat
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

class BgenInputFormatV11 extends IndexedBinaryInputFormat[BgenRecordV11] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    BgenRecordV11] = {
    reporter.setStatus(split.toString)
    new BgenBlockReaderV11(job, split.asInstanceOf[FileSplit])
  }
}

class BgenInputFormatV12 extends IndexedBinaryInputFormat[BgenRecordV12] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    BgenRecordV12] = {
    reporter.setStatus(split.toString)
    new BgenBlockReaderV12(job, split.asInstanceOf[FileSplit])
  }
}
