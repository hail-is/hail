package org.broadinstitute.hail.rest

import org.apache.log4j.{Logger, Level}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

object SparkStuff {
  val conf = new SparkConf()
    .setAppName("hail-t2d-api")
    .setMaster("local[1]")

  conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  conf.set("spark.executor.memory", "2g")
  conf.set("spark.driver.memory", "1g")
  conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")

  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)

  val hadoopConf = sc.hadoopConfiguration
  hadoopConf.set("io.compression.codecs",
    "org.apache.hadoop.io.compress.DefaultCodec,org.broadinstitute.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec")

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
}
