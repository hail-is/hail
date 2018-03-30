package org.apache.spark

import org.apache.spark.executor.InputMetrics

object ExposedMetrics {
  def incrementRecord(metrics: InputMetrics) {
    metrics.incRecordsRead(1)
  }

  def incrementBytes(metrics: InputMetrics, nBytes: Long) {
    metrics.incBytesRead(nBytes)
  }

  def setBytes(metrics: InputMetrics, nBytes: Long) {
    metrics.setBytesRead(nBytes)
  }
}
