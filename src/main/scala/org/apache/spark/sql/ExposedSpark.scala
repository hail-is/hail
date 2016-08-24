package org.apache.spark.sql

object ExposedSpark {
  class ExposedDataFrameReader(sqlContext: SQLContext) extends DataFrameReader(sqlContext)
}
