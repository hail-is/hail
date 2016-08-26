package org.broadinstitute.hail.io

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.broadinstitute.hail.variant._
import scala.collection.mutable.ArrayBuffer


abstract class IndexedBinaryInputFormat[T] extends FileInputFormat[LongWritable, T] {

  def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable, T]
}
