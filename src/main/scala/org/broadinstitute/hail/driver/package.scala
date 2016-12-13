package org.broadinstitute.hail

import java.util
import java.util.Properties

import org.apache.log4j.{LogManager, PropertyConfigurator}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.sql.SQLContext
import org.apache.spark.{ProgressBarBuilder, SparkConf, SparkContext}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.VariantDataset

import scala.collection.JavaConverters._

package object driver {

  case class CountResult(nSamples: Int,
    nVariants: Long,
    nCalled: Option[Long]) {
    def nGenotypes: Long = nSamples * nVariants

    def callRate: Option[Double] =
      nCalled.flatMap(nCalled => divOption[Double](nCalled.toDouble * 100.0, nGenotypes))

    def toJavaMap: util.Map[String, Any] = {
      var m: Map[String, Any] = Map("nSamples" -> nSamples,
        "nVariants" -> nVariants,
        "nGenotypes" -> nGenotypes)
      nCalled.foreach { nCalled => m += "nCalled" -> nCalled }
      callRate.foreach { callRate => m += "callRate" -> callRate }
      m.asJava
    }
  }

  def count(vds: VariantDataset, countGenotypes: Boolean): CountResult = {
    val (nVariants, nCalled) =
      if (countGenotypes) {
        val (nVar, nCalled) = vds.rdd.map { case (v, (va, gs)) =>
          (1L, gs.count(_.isCalled).toLong)
        }.fold((0L, 0L)) { (comb, x) =>
          (comb._1 + x._1, comb._2 + x._2)
        }
        (nVar, Some(nCalled))
      } else
        (vds.nVariants, None)

    CountResult(vds.nSamples, nVariants, nCalled)
  }

  def configureAndCreateSparkContext(appName: String, master: Option[String], local: String = "local[*]",
    logFile: String = "hail.log", quiet: Boolean = false, append: Boolean = false, parquetCompression: String = "uncompressed",
    blockSize: Long = 1L, branchingFactor: Int = 50, tmpDir: String = "/tmp"): SparkContext = {
    require(blockSize >= 0)
    require(branchingFactor > 0)

    HailConfiguration.tmpDir = tmpDir
    HailConfiguration.branchingFactor = branchingFactor

    val conf = new SparkConf().setAppName(appName)

    master match {
      case Some(m) =>
        conf.setMaster(m)
      case None =>
        if (!conf.contains("spark.master"))
          conf.setMaster(local)
    }

    val logProps = new Properties()
    if (quiet) {
      logProps.put("log4j.rootLogger", "OFF, stderr")
      logProps.put("log4j.appender.stderr", "org.apache.log4j.ConsoleAppender")
      logProps.put("log4j.appender.stderr.Target", "System.err")
      logProps.put("log4j.appender.stderr.threshold", "OFF")
      logProps.put("log4j.appender.stderr.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.stderr.layout.ConversionPattern", "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    } else {
      logProps.put("log4j.rootLogger", "INFO, logfile")
      logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
      logProps.put("log4j.appender.logfile.append", append.toString)
      logProps.put("log4j.appender.logfile.file", logFile)
      logProps.put("log4j.appender.logfile.threshold", "INFO")
      logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.logfile.layout.ConversionPattern", "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    }

    LogManager.resetConfiguration()
    PropertyConfigurator.configure(logProps)

    conf.set("spark.ui.showConsoleProgress", "false")

    conf.set(
      "spark.hadoop.io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec," +
        "org.broadinstitute.hail.io.compress.BGzipCodec," +
        "org.apache.hadoop.io.compress.GzipCodec")

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val tera = 1024L * 1024L * 1024L * 1024L

    conf.set("spark.sql.parquet.compression.codec", parquetCompression)
    conf.set("spark.sql.files.openCostInBytes", tera.toString)
    conf.set("spark.sql.files.maxPartitionBytes", tera.toString)

    conf.set("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", (blockSize * 1024L * 1024L).toString)

    /* `DataFrame.write` writes one file per partition.  Without this, read will split files larger than the default
     * parquet block size into multiple partitions.  This causes `OrderedRDD` to fail since the per-partition range
     * no longer line up with the RDD partitions.
     *
     * For reasons we don't understand, the DataFrame code uses `SparkHadoopUtil.get.conf` instead of the Hadoop
     * configuration in the SparkContext.  Set both for consistency.
     */
    SparkHadoopUtil.get.conf.setLong("parquet.block.size", tera)
    conf.set("spark.hadoop.parquet.block.size", tera.toString)

    // load additional Spark properties from HAIL_SPARK_PROPERTIES
    val hailSparkProperties = System.getenv("HAIL_SPARK_PROPERTIES")
    if (hailSparkProperties != null) {
      hailSparkProperties
        .split(",")
        .foreach { p =>
          p.split("=") match {
            case Array(k, v) =>
              log.info(s"set Spark property from HAIL_SPARK_PROPERTIES: $k=$v")
              conf.set(k, v)
            case _ =>
              warn(s"invalid key-value property pair in HAIL_SPARK_PROPERTIES: $p")
          }
        }
    }

    log.info(s"Spark properties: ${
      conf.getAll.map { case (k, v) =>
        s"$k=$v"
      }.mkString(", ")
    }")

    val sc = new SparkContext(conf)
    ProgressBarBuilder.build(sc)
    sc
  }

  def createSQLContext(sc: SparkContext): SQLContext = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    sc.getConf.getAll.foreach { case (k, v) =>
      if (k.startsWith("spark.sql."))
        sqlContext.setConf(k, v)
    }
    sqlContext
  }
}
