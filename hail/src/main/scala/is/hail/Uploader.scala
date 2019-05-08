package is.hail

import java.io.{PrintWriter, StringWriter}
import java.net.InetAddress
import java.util.concurrent.LinkedBlockingQueue

import is.hail.expr.ir.{BaseIR, Pretty}
import is.hail.utils._
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClientBuilder
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager
import org.apache.http.util.EntityUtils
import org.json4s.JsonAST.{JInt, JNull, JObject, JString}
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._

object Uploader {
  private lazy val theUploader: Uploader = new Uploader

  var url = "https://upload.hail.is/upload"

  var uploadEnabled: Boolean = false

  var email: String = _

  def uploadPipeline(ir0: BaseIR, ir: BaseIR) {
    if (!uploadEnabled)
      return

    // for stack trace
    val e = new Traceback

    val w = new StringWriter
    // closes w
    val stackTrace = using(new PrintWriter(w)) { p =>
      e.printStackTrace(p)
      w.toString
    }

    val contents =
      s"ir0:\n${ Pretty(ir0) }\n\nir:\n${ Pretty(ir) }\n\nfrom:\n${ stackTrace }"

    theUploader.enqueueUpload("ir", contents, email)
  }

  def upload(typ: String, contents: String) {
    theUploader.upload(typ, contents, email)
  }
}

class Uploader { self =>

  private val config = {
    val hc = HailContext.get
    val sc = hc.sc
    val fs = hc.sFS
    val runtime = Runtime.getRuntime

    JObject(
      "hostname" -> JString(InetAddress.getLocalHost.getHostName),
      "jvm_version" -> JString(System.getProperty("java.version")),
      "jvm_properties" -> JObject(
        System.getProperties.asScala.map { case (k, v) =>
          k -> JString(v)
        }
          .toList),
      "jvm_runtime" -> JObject(
        "available_processors" -> JInt(runtime.availableProcessors()),
        "total_memory" -> JInt(runtime.totalMemory()),
        "max_memory" -> JInt(runtime.maxMemory())),
      "spark_version" -> JString(hc.sc.version),
      "spark_conf" -> JObject(
        sc.getConf.getAll.map { case (k, v) =>
          k -> JString(v)
        }
          .toList),
      "hadoop_conf" -> JObject(
        fs.getProperties.map { entry =>
          entry.getKey -> JString(entry.getValue)
        }
          .toList),
      "hail_version" -> JString(hc.version))
  }

  private val httpClient = HttpClientBuilder.create()
    .setConnectionManager(new PoolingHttpClientConnectionManager())
    .build()

  private val queue = new LinkedBlockingQueue[(String, String, String)]()

  val t = new Thread {
    override def run() {
      while (true) {
        val item = queue.take()
        if (item == null)
          return

        val (typ, contents, email) = item

        try {
          self.upload(typ, contents, email)
        } catch {
          case e: Exception =>
            log.warn(s"upload failed, caught $e")
        }
      }
    }
  }

  t.start()

  def upload(typ: String, contents: String, email: String) {
    val request = new HttpPost(Uploader.url)

    val jEmail = if (email != null) JString(email) else JNull
    val jv = JObject(
      "type" -> JString(typ),
      "email" -> jEmail,
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

  def enqueueUpload(typ: String, contents: String, email: String) {
    queue.offer((typ, contents, email))
  }

  def join() {
    queue.add(null)
    t.join()
  }
}
