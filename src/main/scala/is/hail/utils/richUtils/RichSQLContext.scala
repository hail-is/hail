package is.hail.utils.richUtils

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.functions._
import is.hail.utils._

class RichSQLContext(val sqlContext: SQLContext) extends AnyVal {
  def readParquetSorted(dirname: String, selection: Option[Array[String]] = None): RDD[Row] = {
    val parquetFiles = sqlContext.sparkContext.hadoopConfiguration.globAll(Array(dirname + "/*.parquet"))
    if (parquetFiles.isEmpty)
      return sqlContext.sparkContext.emptyRDD[Row]

    var df = sqlContext.read.parquet(dirname + "/part-*")
    selection.foreach { cols =>
      df = df.select(cols.map(col): _*)
    }

    val rdd = df.rdd

    val oldIndices = rdd.partitions
      .map { p => getParquetPartNumber(partitionPath(p)) }
      .zipWithIndex
      .sortBy(_._1)
      .map(_._2)

    rdd.reorderPartitions(oldIndices)
  }
}
