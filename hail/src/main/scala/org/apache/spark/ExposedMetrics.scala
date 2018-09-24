package org.apache.spark

import org.apache.spark.executor.{InputMetrics, OutputMetrics}

object ExposedMetrics {
  def incrementRecord(metrics: InputMetrics) {
    metrics.incRecordsRead(1)
  }

  def incrementBytes(metrics: InputMetrics, nBytes: Long) {
    metrics.incBytesRead(nBytes)
  }

  def setBytes(metrics: OutputMetrics, nBytes: Long) {
    metrics.setBytesWritten(nBytes)
  }

  def setRecords(metrics: OutputMetrics, nRecords: Long) {
    metrics.setRecordsWritten(nRecords)
  }
}
