package org.broadinstitute.hail.sparkextras

import org.apache.hadoop.fs.Path
import org.apache.spark.sql.ExposedSpark.ExposedDataFrameReader
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.sql.execution.datasources.parquet.PartitionedParquetRelation
import org.apache.spark.sql.{DataFrame, DataFrameReader, SQLContext}

/**
  * Copied and slightly modified from org.apache.spark.sql.DataFrameReader, version 1.5.0
  * Changed to call PartitionedParquetRelation instead of ParquetRelation
  */
class PartitionedDataFrameReader(sqlContext: SQLContext) extends ExposedDataFrameReader(sqlContext) {
  override def parquet(paths: String*): DataFrame = {
    if (paths.isEmpty) {
      sqlContext.emptyDataFrame
    } else {
      val globbedPaths = paths.flatMap { path =>
        val hdfsPath = new Path(path)
        val fs = hdfsPath.getFileSystem(sqlContext.sparkContext.hadoopConfiguration)
        val qualified = hdfsPath.makeQualified(fs.getUri, fs.getWorkingDirectory)
        SparkHadoopUtil.get.globPathIfNecessary(qualified)
      }.toArray

      sqlContext.baseRelationToDataFrame(
        new PartitionedParquetRelation(
          globbedPaths.map(_.toString), None, None, None, Map.empty[String, String])(sqlContext))
    }
  }
}
