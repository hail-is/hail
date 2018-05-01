package is.hail.io

import is.hail.HailContext
import is.hail.table.Table
import org.elasticsearch.spark.sql._

import scala.collection.Map

object ElasticsearchConnector {

  def export(t: Table, host: String = "localhost", port: Int = 9200,
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

    val df = t
      .expandTypes()
      .toDF(HailContext.get.sqlContext)
      .saveToEs(s"${ index }/${ indexType }", mergedConfig)
  }
}
