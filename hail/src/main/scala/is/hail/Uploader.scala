package is.hail

import is.hail.utils._
import java.util.concurrent.{LinkedBlockingQueue}

import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClientBuilder
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager
import org.apache.http.util.EntityUtils
import org.json4s.JsonAST.{JInt, JObject, JString}
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._

object Uploader {
  private lazy val theUploader: Uploader = new Uploader

  def enqueueUpload(typ: String, contents: String) {
    theUploader.enqueueUpload(typ, contents)
  }

  def upload(typ: String, contents: String) {
    theUploader.upload(typ, contents)
  }
}

class Uploader {
  self =>

  private val config = {
    val hc = HailContext.get
    val sc = hc.sc
    val hConf = hc.hadoopConf
    val runtime = Runtime.getRuntime

    JObject(
      "jvm_version" -> JString(System.getProperty("java.version")),
      "jvm_properties" -> JObject(
        System.getProperties.asScala.map { case (k, v) =>
          k -> JString(v)
        }
          .toList),
      "jvm_runtime" -> JObject(
        "available_processors" -> JInt(runtime.availableProcessors()),
        "free_memory" -> JInt(runtime.freeMemory()),
        "total_memory" -> JInt(runtime.totalMemory()),
        "max_memory" -> JInt(runtime.maxMemory())),
      "spark_version" -> JString(hc.sc.version),
      "spark_conf" -> JObject(
        sc.getConf.getAll.map { case (k, v) =>
          k -> JString(v)
        }
          .toList),
      "hadoop_conf" -> JObject(
        hConf.iterator().asScala.map { entry =>
          entry.getKey -> JString(entry.getValue)
        }
          .toList),
      "hail_version" -> JString(hc.version))
  }

  private val httpClient = HttpClientBuilder.create()
    .setConnectionManager(new PoolingHttpClientConnectionManager())
    .build()

  private val queue = new LinkedBlockingQueue[(String, String)]()

  val t = new Thread {
    override def run() {
      while (true) {
        val item = queue.take()
        if (item == null)
          return

        val (typ, contents) = item
        self.upload(typ, contents)
      }
    }
  }

  t.start()

  def upload(typ: String, contents: String) {
    // FIXME upload.hail.is/upload
    val request = new HttpPost("https://test.hail.is/upload")

    val jv = JObject(
      "type" -> JString(typ),
      "config" -> config,
      "contents" -> JString(contents))

    val data = new StringEntity(
      JsonMethods.compact(JsonMethods.render(jv)))
    request.addHeader("content-type", "application/json")
    request.setEntity(data)

    using(httpClient.execute(request)) { response =>
      EntityUtils.consume(response.getEntity)
    }
  }

  def enqueueUpload(typ: String, contents: String) {
    queue.offer((typ, contents))
  }

  def join() {
    queue.add(null)
    t.join()
  }
}
