package is.hail.io

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

abstract class IndexedBinaryInputFormat[T] extends FileInputFormat[LongWritable, T] {

  def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter)
    : RecordReader[LongWritable, T]
}
