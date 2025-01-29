package is.hail.utils.richUtils

import is.hail.utils._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class RichSparkContext(val sc: SparkContext) extends AnyVal {
  def textFilesLines(files: Array[String], nPartitions: Int = sc.defaultMinPartitions)
    : RDD[WithContext[String]] = {

    /* Don't use:
     * sc.union(files.map(sc.textFile, nPartitions)) since it asks for nPartitions per file instead
     * of nPartitions over all. */
    val rdd = sc.textFile(files.mkString(","), nPartitions)
    val partitionFile = rdd.partitions.map(partitionPath)

    rdd
      .mapPartitionsWithIndex { case (i, it) =>
        // FIXME subclass TextInputFormat to return (file, line)
        val file = partitionFile(i)
        it.map(line => WithContext(line, Context(line, file, None)))
      }
  }

  def textFileLines(file: String, nPartitions: Int = sc.defaultMinPartitions)
    : RDD[WithContext[String]] =
    sc.textFile(file, nPartitions)
      .map(l => WithContext(l, Context(l, file, None)))
}
