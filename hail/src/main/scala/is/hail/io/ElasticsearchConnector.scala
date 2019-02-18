package is.hail.io

import is.hail.expr.ir.{Interpret, TableIR}
import org.elasticsearch.spark.sql._

import collection.JavaConverters._
import scala.collection.Map

object ElasticsearchConnector {

  def export(
    tir: TableIR,
    host: String,
    port: Int,
    index: String,
    indexType: String,
    blockSize: Int,
    config: java.util.HashMap[String, String],
    verbose: Boolean) {
    export(tir, host, port, index, indexType, blockSize,
      Option(config).map(_.asScala.toMap).getOrElse(Map.empty[String, String]), verbose)
  }

  def export(tir: TableIR, host: String = "localhost", port: Int = 9200,
    index: String, indexType: String, blockSize: Int = 1000,
    config: Map[String, String], verbose: Boolean = true) {

    // config docs: https://www.elastic.co/guide/en/elasticsearch/hadoop/master/configuration.html

    val defaultConfig = Map(
      "es.nodes" -> host,
      "es.port" -> port.toString,
      "es.batch.size.entries" -> blockSize.toString,
      "es.index.auto.create" -> "true")

    val mergedConfig = if (config == null)
      defaultConfig
    else
      defaultConfig ++ config

    if (verbose)
      println(s"Config ${ mergedConfig }")

    val df = Interpret(tir)
      .toDF()
      .saveToEs(s"${ index }/${ indexType }", mergedConfig)
  }
}
