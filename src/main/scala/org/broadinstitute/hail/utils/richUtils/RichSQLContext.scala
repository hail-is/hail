package org.broadinstitute.hail.utils.richUtils

import java.lang.reflect.Method

import org.apache.hadoop.fs.{Path, PathIOException}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext, SparkExport}
import org.apache.spark.sql.functions._
import org.broadinstitute.hail.utils._

class RichSQLContext(val sqlContext: SQLContext) extends AnyVal {
  def readParquetSorted(dirname: String, selection: Option[Array[String]] = None): RDD[Row] = {
    val parquetFiles = sqlContext.sparkContext.hadoopConfiguration.globAll(Array(dirname + "/*.parquet"))
    if (parquetFiles.isEmpty)
      return sqlContext.sparkContext.emptyRDD[Row]

    var df = sqlContext.read.parquet(dirname + "/part-r-*")
    selection.foreach { cols =>
      df = df.select(cols.map(col): _*)
    }

    val rdd = df.rdd

    val oldIndices = rdd.partitions
      .map { p =>
        val parquetInputSplit = SparkExport.sqlNewHadoopPartitionRawSplit(p)

        /*
         * Use reflection to invoke getPath instead of importing ParquetInputSplit and casting since the parquet
         * package (parquet vs org.apache.parquet) moved and depends on the distribution we are running against.
         */
        def f(c: Class[_], method: String): Method = {
          try {
            c.getDeclaredMethod(method)
          } catch {
            case _: Exception =>
              assert(c != classOf[java.lang.Object])
              f(c.getSuperclass, method)
          }
        }
        val m = f(parquetInputSplit.getClass, "getPath")
        val path = m.invoke(parquetInputSplit).asInstanceOf[Path]
        getParquetPartNumber(path.getName)
      }
      .zipWithIndex
      .sortBy(_._1)
      .map(_._2)

    rdd.reorderPartitions(oldIndices)
  }
}
