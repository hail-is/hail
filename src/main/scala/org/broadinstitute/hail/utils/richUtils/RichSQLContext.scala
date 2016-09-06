package org.broadinstitute.hail.utils.richUtils

import org.apache.hadoop.fs.PathIOException
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.broadinstitute.hail.utils._

class RichSQLContext(val sqlContext: SQLContext) extends AnyVal {
  def sortedParquetRead(dirname: String): Option[DataFrame] = {
    val partRegex = ".*/?part-r-(\\d+)-.*\\.parquet.*".r
    def getPartNumber(fname: String): Int = {
      fname match {
        case partRegex(i) => i.toInt
        case _ => throw new PathIOException(s"invalid parquet file `$fname'")
      }
    }

    val parquetFiles = sqlContext.sparkContext.hadoopConfiguration.globAll(Array(s"""$dirname/*.parquet"""))
      .sortBy(getPartNumber)

    if (parquetFiles.isEmpty)
      None
    else Some(sqlContext.read.parquet(parquetFiles: _*))
  }
}

