package org.apache.spark.sql

import org.apache.hadoop.mapreduce.InputSplit
import org.apache.spark.Partition
import org.apache.spark.rdd.SqlNewHadoopPartition

object SparkExport {
  def sqlNewHadoopPartitionRawSplit(p: Partition): InputSplit = p.asInstanceOf[SqlNewHadoopPartition].serializableHadoopSplit.value
}
