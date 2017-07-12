package is.hail.io

import is.hail.keytable.KeyTable
import org.elasticsearch.spark.sql._
import scala.collection.Map

object ElasticsearchConnector {

  def export(kt: KeyTable, host: String = "localhost", port: Int = 9200,
             index: String, indexType: String, blockSize: Int = 1000, verbose: Boolean = true) {

    // config docs: https://www.elastic.co/guide/en/elasticsearch/hadoop/master/configuration.html
    var config = Map(
      "es.nodes" -> host,
      "es.port" -> port.toString,
      "es.batch.size.entries" -> blockSize.toString,
      "es.index.auto.create" -> "true"
    )
    // other potentially useful settings:
    // es.write.operation // default: index (create, update, upsert)
    // es.http.timeout // default 1m
    // es.http.retries // default 3
    // es.batch.size.bytes  // default 1mb
    // es.batch.size.entries  // default 1000
    // es.batch.write.refresh // default true  (Whether to invoke an index refresh or not after a bulk update has been completed)

    val df = kt.toDF(kt.hc.sqlContext)
    if (verbose) {
      println(s"Exporting ${df.count()} rows to ${host}:${port}/${index}/${indexType}")
      df.printSchema()
    }
    df.saveToEs(s"${index}/${indexType}", config)
  }
}
