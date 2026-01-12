package is.hail.sparkextras.implicits

import is.hail.utils._

import org.apache.hadoop.mapred.FileSplit
import org.apache.hadoop.mapreduce.lib.input.{FileSplit => NewFileSplit}
import org.apache.spark.{Partition, SparkContext}
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

  /* Use reflection to get the path of a partition coming from a Parquet read. This requires
   * accessing Spark internal interfaces. It works with Spark 1 and 2 and doesn't depend on the
   * location of the Parquet package (parquet vs org.apache.parquet) which can vary between
   * distributions. */
  def partitionPath(p: Partition): String = {
    p.getClass.getCanonicalName match {
      case "org.apache.spark.rdd.SqlNewHadoopPartition" =>
        val split = invokeMethod(invokeMethod(p, "serializableHadoopSplit"), "value").asInstanceOf[
          NewFileSplit
        ]
        split.getPath.getName

      case "org.apache.spark.sql.execution.datasources.FilePartition" =>
        val files = invokeMethod(p, "files").asInstanceOf[Seq[_ <: AnyRef]]
        assert(files.length == 1)
        invokeMethod(files.head, "filePath").asInstanceOf[String]

      case "org.apache.spark.rdd.HadoopPartition" =>
        val split = invokeMethod(invokeMethod(p, "inputSplit"), "value").asInstanceOf[FileSplit]
        split.getPath.getName
    }
  }
}
