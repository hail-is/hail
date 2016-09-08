package org.broadinstitute.hail.utils.richUtils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._

class RichSparkContext(val sc: SparkContext) extends AnyVal {
  def textFilesLines(files: Array[String], f: String => Unit = s => (),
    nPartitions: Int = sc.defaultMinPartitions): RDD[WithContext[String]] = {
    files.foreach(f)
    sc.union(
      files.map(file =>
        sc.textFileLines(file, nPartitions)))
  }

  def textFileLines(file: String, nPartitions: Int = sc.defaultMinPartitions): RDD[WithContext[String]] =
    sc.textFile(file, nPartitions)
      .map(l => WithContext(l, TextContext(l, file, None)))
}
