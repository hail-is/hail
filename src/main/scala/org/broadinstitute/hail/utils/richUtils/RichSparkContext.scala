package org.broadinstitute.hail.utils.richUtils

import org.apache.hadoop.mapred.FileSplit
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkExport
import org.broadinstitute.hail.utils._

class RichSparkContext(val sc: SparkContext) extends AnyVal {
  def textFilesLines(files: Array[String],
    nPartitions: Int = sc.defaultMinPartitions): RDD[WithContext[String]] = {

    /*
     * Don't use:
     *   sc.union(files.map(sc.textFile, nPartitions))
     * since it asks for nPartitions per file instead of nPartitions over all.
     */
    val rdd = sc.textFile(files.mkString(","), nPartitions)
    val partitionsBc = sc.broadcast(rdd.partitions)

    rdd
      .mapPartitionsWithIndex { case (i, it) =>
        // FIXME subclass TextInputFormat to return (file, line)
        val file = SparkExport.hadoopPartitionSplit(partitionsBc.value(i)).asInstanceOf[FileSplit].getPath.toString
        it.map { line =>
          WithContext(line, TextContext(line, file, None))
        }
      }
  }

  def textFileLines(file: String, nPartitions: Int = sc.defaultMinPartitions): RDD[WithContext[String]] =
    sc.textFile(file, nPartitions)
      .map(l => WithContext(l, TextContext(l, file, None)))
}
