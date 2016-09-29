package org.broadinstitute.hail.io.plink

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.broadinstitute.hail.io.{IndexedBinaryInputFormat, KeySerializedValueRecord}

class PlinkInputFormat extends IndexedBinaryInputFormat[PlinkRecord] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    PlinkRecord] = {
    reporter.setStatus(split.toString)
    new PlinkBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
