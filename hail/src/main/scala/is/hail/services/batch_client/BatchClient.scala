package is.hail.services.batch_client

import java.nio.charset.StandardCharsets

import is.hail.utils._
import is.hail.services._
import is.hail.services.{DeployConfig, Tokens}
import org.apache.commons.io.IOUtils
import org.apache.http.{HttpEntity, HttpEntityEnclosingRequest}
import org.apache.http.client.methods.{HttpDelete, HttpGet, HttpPatch, HttpPost, HttpUriRequest}
import org.apache.http.entity.{ByteArrayEntity, ContentType, StringEntity}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.util.EntityUtils
import org.apache.log4j.{LogManager, Logger}
import org.json4s.{DefaultFormats, Formats, JObject, JValue}
import org.json4s.jackson.JsonMethods

import scala.util.Random

class NoBodyException(message: String, cause: Throwable) extends Exception(message, cause) {
  def this() = this(null, null)

  def this(message: String) = this(message, null)
}

object BatchClient {
  lazy val log: Logger = LogManager.getLogger("BatchClient")

  def fromSessionID(sessionID: String): BatchClient = {
    val deployConfig = DeployConfig.get
    new BatchClient(deployConfig,
      new Tokens(Map(
        deployConfig.getServiceNamespace("batch") -> sessionID)))
  }
}

class BatchClient(
  deployConfig: DeployConfig,
  requester: Requester
) {
  def this() = this(DeployConfig.get, new Requester("batch"))

  def this(deployConfig: DeployConfig) = this(deployConfig, new Requester("batch"))

  def this(tokens: Tokens) = this(DeployConfig.get, new Requester(tokens, "batch"))

  def this(deployConfig: DeployConfig, tokens: Tokens) =
    this(deployConfig, new Requester(tokens, "batch"))

  import BatchClient._
  import requester.request

  private[this] val baseUrl = deployConfig.baseUrl("batch")

  def get(path: String): JValue =
    request(new HttpGet(s"$baseUrl$path"))

  def post(path: String, body: HttpEntity): JValue =
    request(new HttpPost(s"$baseUrl$path"), body = body)

  def post(path: String, json: JValue = null): JValue =
    post(path,
      if (json != null)
        new StringEntity(
          JsonMethods.compact(json),
          ContentType.create("application/json"))
      else
        null)

  def patch(path: String): JValue =
    request(new HttpPatch(s"$baseUrl$path"))

  def delete(path: String, token: String): JValue =
    request(new HttpDelete(s"$baseUrl$path"))

  def createJobs(batchID: Long, jobs: IndexedSeq[JObject]): Unit = {
    val bunches = new ArrayBuilder[Array[Array[Byte]]]()

    val bunchb = new ArrayBuilder[Array[Byte]]()

    var i = 0
    var size = 0
    while (i < jobs.length) {
      val jobBytes = JsonMethods.compact(jobs(i)).getBytes(StandardCharsets.UTF_8)
      if (size + jobBytes.length > 1024 * 1024) {
        bunches += bunchb.result()
        bunchb.clear()
        size = 0
      }
      bunchb += jobBytes
      size += jobBytes.length
      i += 1
    }
    assert(bunchb.size > 0)

    bunches += bunchb.result()
    bunchb.clear()
    size = 0

    val b = new ArrayBuilder[Byte]()

    i = 0 // reuse
    while (i < bunches.length) {
      val bunch = bunches(i)
      b += '['
      var j = 0
      while (j < bunch.length) {
        if (j > 0)
           b += ','
        b ++= bunch(j)
        j += 1
      }
      b += ']'
      val data = b.result()
      post(
        s"/api/v1alpha/batches/$batchID/jobs/create",
        new ByteArrayEntity(
          data,
          ContentType.create("application/json")))
      b.clear()
      i += 1
    }
  }

  def waitForBatch(batchID: Long): JValue = {
    implicit val formats: Formats = DefaultFormats

    val start = System.nanoTime()

    while (true) {
      val batch = get(s"/api/v1alpha/batches/$batchID")
      if ((batch \ "complete").extract[Boolean])
        return batch

      // wait 10% of duration so far
      // at least, 50ms
      // at most, 5s
      val now = System.nanoTime()
      val elapsed = now - start
      var d = math.max(
        math.min(
          (0.1 * (0.8 + Random.nextFloat() * 0.4) * (elapsed / 1000.0 / 1000)).toInt,
          5000),
        50)
      Thread.sleep(d)
    }

    throw new AssertionError("unreachable")
  }

  def run(batchJson: JObject, jobs: IndexedSeq[JObject]): JValue = {
    val resp = post("/api/v1alpha/batches/create", json = batchJson)

    implicit val formats: Formats = DefaultFormats
    val batchID = (resp \ "id").extract[Long]

    log.info(s"run: created batch $batchID")

    createJobs(batchID, jobs)

    patch(s"/api/v1alpha/batches/$batchID/close")

    waitForBatch(batchID)
  }
}
