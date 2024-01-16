package org.apache.spark

import org.apache.spark.executor.{InputMetrics, OutputMetrics}

object ExposedMetrics {
  def incrementRecord(metrics: InputMetrics): Unit = {
    metrics.incRecordsRead(1)
  }

  def incrementBytes(metrics: InputMetrics, nBytes: Long): Unit = {
    metrics.incBytesRead(nBytes)
  }

  def setBytes(metrics: OutputMetrics, nBytes: Long): Unit = {
    metrics.setBytesWritten(nBytes)
  }

  def setRecords(metrics: OutputMetrics, nRecords: Long): Unit = {
    metrics.setRecordsWritten(nRecords)
  }
}
