package is.hail.io

import scala.collection.JavaConverters._
import scala.collection.Map

import org.apache.spark
import org.elasticsearch.spark.sql._

object ElasticsearchConnector {

  def export(
    df: spark.sql.DataFrame,
    host: String,
    port: Int,
    index: String,
    indexType: String,
    blockSize: Int,
    config: java.util.HashMap[String, String],
    verbose: Boolean,
  ) {
    export(
      df,
      host,
      port,
      index,
      indexType,
      blockSize,
      Option(config).map(_.asScala.toMap).getOrElse(Map.empty[String, String]),
      verbose,
    )
  }

  def export(
    df: spark.sql.DataFrame,
    host: String = "localhost",
    port: Int = 9200,
    index: String,
    indexType: String,
    blockSize: Int = 1000,
    config: Map[String, String],
    verbose: Boolean = true,
  ) {

    // config docs: https://www.elastic.co/guide/en/elasticsearch/hadoop/master/configuration.html

    val defaultConfig = Map(
      "es.nodes" -> host,
      "es.port" -> port.toString,
      "es.batch.size.entries" -> blockSize.toString,
      "es.index.auto.create" -> "true",
    )

    val mergedConfig = if (config == null)
      defaultConfig
    else
      defaultConfig ++ config

    if (verbose)
      println(s"Config $mergedConfig")

    df.saveToEs(s"$index/$indexType", mergedConfig)
  }
}
