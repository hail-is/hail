package is.hail.io

import is.hail.keytable.KeyTable
import org.elasticsearch.spark.sql._
import scala.collection.Map
import scala.collection.mutable

object ElasticsearchConnector {

  def export(kt: KeyTable, host: String = "localhost", port: Int = 9200,
             index: String, indexType: String, blockSize: Int = 1000,
             config: Map[String, String], verbose: Boolean = true) {

    // config docs: https://www.elastic.co/guide/en/elasticsearch/hadoop/master/configuration.html

    val defaultConfig = Map(
      "es.nodes" -> host,
      "es.port" -> port.toString,
      "es.batch.size.entries" -> blockSize.toString,
      "es.index.auto.create" -> "true"
    )

    val mergedConfig = if (config == null)
        defaultConfig
      else
        (defaultConfig.keySet | config.keySet).map (key => key -> config.getOrElse(key, defaultConfig.getOrElse(key, "").toString())).toMap

    if (verbose)
      println(s"Config ${mergedConfig}")

    val df = kt.toDF(kt.hc.sqlContext)
    if (verbose)
      println(s"Exporting ${df.count()} rows to ${host}:${port}/${index}/${indexType}")

    df.saveToEs(s"${index}/${indexType}", mergedConfig)
  }
}
