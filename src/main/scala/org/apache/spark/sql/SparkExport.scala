package org.apache.spark.sql

import org.apache.hadoop.mapreduce.{InputSplit => NewInputSplit}
import org.apache.hadoop.mapred.InputSplit
import org.apache.spark.Partition
import org.apache.spark.rdd.{HadoopPartition, SqlNewHadoopPartition}

object SparkExport {
  def sqlNewHadoopPartitionRawSplit(p: Partition): NewInputSplit = p.asInstanceOf[SqlNewHadoopPartition].serializableHadoopSplit.value
  def hadoopPartitionSplit(p: Partition): InputSplit = p.asInstanceOf[HadoopPartition].inputSplit.value
}
