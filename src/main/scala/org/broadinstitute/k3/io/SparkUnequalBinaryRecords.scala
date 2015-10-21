package org.broadinstitute.k3.io

import org.apache.hadoop.conf._
import org.apache.spark.SparkContext
import org.apache.spark.rdd._

object  SparkUnequalBinaryRecords {

  def apply(sc: SparkContext, conf: Configuration, recordPositions: Array[Long]): RDD[Array[Byte]] = {

    null
  }
}
