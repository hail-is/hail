package org.broadinstitute.hail.io.plink

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.broadinstitute.hail.io.{IndexedBinaryInputFormat, VariantRecord}

class PlinkInputFormat extends IndexedBinaryInputFormat[Int] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    VariantRecord[Int]] = {
    reporter.setStatus(split.toString)
    new PlinkBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
