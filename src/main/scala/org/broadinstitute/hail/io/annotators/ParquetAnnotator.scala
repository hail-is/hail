package org.broadinstitute.hail.io.annotators

import org.apache.spark.SparkEnv
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.variant._

object ParquetAnnotator {
  def apply(filename: String, sqlContext: SQLContext): (RDD[(Variant, Annotation)], expr.Type) = {

    fatalIf(!filename.endsWith(".faf"), "expected a file ending in '.faf'")

    val signature = readDataFile(filename + "/signature.ser",
      sqlContext.sparkContext.hadoopConfiguration) {
      dis => {
        val serializer = SparkEnv.get.serializer.newInstance()
        serializer.deserializeStream(dis).readObject[expr.Type]
      }
    }

    val df = sqlContext.read.parquet(filename + "/rdd.parquet")
    (df.rdd.map(row => (Variant.fromRow(row.getAs[Row](0)), row.get(1))), signature)
  }
}
