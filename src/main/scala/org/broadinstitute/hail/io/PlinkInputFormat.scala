package org.broadinstitute.hail.io

import org.apache.hadoop.mapred._

class PlinkInputFormat extends IndexedBinaryInputFormat {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[RichLongWritable,
    ByteBlock] = {
    reporter.setStatus(split.toString)
    new PlinkBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
