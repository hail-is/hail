package org.broadinstitute.hail.io

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

class PlinkInputFormat extends IndexedBinaryInputFormat[Int] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    ParsedLine[Int]] = {
    reporter.setStatus(split.toString)
    new PlinkBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
