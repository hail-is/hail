package is.hail.io

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import is.hail.variant._
import scala.collection.mutable.ArrayBuffer


abstract class IndexedBinaryInputFormat[T] extends FileInputFormat[LongWritable, T] {

  def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable, T]
}
