package org.broadinstitute.hail.io.bgen

import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.broadinstitute.hail.io.{IndexedBinaryInputFormat, KeySerializedValueRecord}
import org.broadinstitute.hail.variant.Variant

class BgenInputFormat extends IndexedBinaryInputFormat[BgenRecord] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    BgenRecord] = {
    reporter.setStatus(split.toString)
    new BgenBlockReader(job, split.asInstanceOf[FileSplit])
  }
}
