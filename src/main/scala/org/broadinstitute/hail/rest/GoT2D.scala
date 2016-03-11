package org.broadinstitute.hail.rest

import java.net.URI
import org.apache.log4j.{Logger, Level}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.{SparkContext, SparkConf}
import scala.io.Source
import org.apache.hadoop

object SparkStuff {
  val conf = new SparkConf()
    .setAppName("hail-t2d-api")
    .setMaster("local[*]")

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

object GoT2D {
  import SparkStuff._

  def importPValues() {
    val markerIdRegex = """([^:]*):([^_]*)_([^/]*)/(.*)""".r
    val results: RDD[Stat] = sc.textFile(Conf.epactsFile)
      .filter(line => !line.isEmpty && line(0) != '#')
      .map { line =>
        val fields = line.split("\t")
        val markerIdRegex(chrom, pos, ref, alt) = fields(3)
        val pValue = fields(8)
        Stat(chrom, pos.toInt, ref, alt,
          if (pValue == "NA")
            None
          else
            Some(pValue.toDouble))
      } // .repartition(16)

    val hFS = hadoop.fs.FileSystem.get(new URI(Conf.pValuesFile), sc.hadoopConfiguration)
    hFS.delete(new hadoop.fs.Path(Conf.pValuesFile), true)

    import sqlContext.implicits._
    results.toDF()
      .write
      .partitionBy("chrom")
      .parquet(Conf.pValuesFile)
  }

  def results: DataFrame =
    sqlContext.read.parquet(Conf.pValuesFile)
      .cache()
}
