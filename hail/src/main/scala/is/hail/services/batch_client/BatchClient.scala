package is.hail.services.batch_client

import is.hail.expr.ir.ByteArrayBuilder

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
import org.json4s.{DefaultFormats, Formats, JInt, JObject, JString, JValue}
import org.json4s.jackson.JsonMethods

import scala.util.Random

class NoBodyException(message: String, cause: Throwable) extends Exception(message, cause) {
  def this() = this(null, null)

  def this(message: String) = this(message, null)
}

object BatchClient {
  lazy val log: Logger = LogManager.getLogger("BatchClient")
}

class BatchClient(
  deployConfig: DeployConfig,
  requester: Requester
) {

  def this(credentialsPath: String) = this(DeployConfig.get, Requester.fromCredentialsFile(credentialsPath))

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

  def update(batchID: Long, token: String, jobs: IndexedSeq[JObject]) = {
    implicit val formats: Formats = DefaultFormats

    val updateJson = JObject("n_jobs" -> JInt(jobs.length), "token" -> JString(token))
    val bunches = createBunches(jobs)
    val updateID = if (bunches.length == 1) {
      val b = new ByteArrayBuilder()
      b ++= "{\"bunch\":".getBytes(StandardCharsets.UTF_8)
      addBunchBytes(b, bunches(0))
      b ++= ",\"update\":".getBytes(StandardCharsets.UTF_8)
      b ++= JsonMethods.compact(updateJson).getBytes(StandardCharsets.UTF_8)
      b += '}'
      val data = b.result()
      val resp = retryTransientErrors{
        post(s"/api/v1alpha/batches/$batchID/update-fast",
          new ByteArrayEntity(data, ContentType.create("application/json")))
      }
      b.clear()
      (resp \ "update_id").extract[Long]
    } else {
      val resp = retryTransientErrors { post(s"/api/v1alpha/batches/$batchID/updates/create", json = updateJson) }
      val updateID = (resp \ "update_id").extract[Long]

      val b = new ByteArrayBuilder()
      var i = 0
      while (i < bunches.length) {
        addBunchBytes(b, bunches(i))
        val data = b.result()
        retryTransientErrors {
          post(
            s"/api/v1alpha/batches/$batchID/updates/$updateID/jobs/create",
            new ByteArrayEntity(
              data,
              ContentType.create("application/json")))
        }
        b.clear()
        i += 1
      }

      retryTransientErrors { patch(s"/api/v1alpha/batches/$batchID/updates/$updateID/commit") }
      updateID
    }
    log.info(s"run: created update $updateID for batch $batchID")
  }

  def create(batchJson: JObject, jobs: IndexedSeq[JObject]): Long = {
    implicit val formats: Formats = DefaultFormats

    val bunches = createBunches(jobs)
    val batchID = if (bunches.length == 1) {
      val bunch = bunches(0)
      val b = new ByteArrayBuilder()
      b ++= "{\"bunch\":".getBytes(StandardCharsets.UTF_8)
      addBunchBytes(b, bunch)
      b ++= ",\"batch\":".getBytes(StandardCharsets.UTF_8)
      b ++= JsonMethods.compact(batchJson).getBytes(StandardCharsets.UTF_8)
      b += '}'
      val data = b.result()
      val resp = retryTransientErrors{
        post("/api/v1alpha/batches/create-fast",
          new ByteArrayEntity(data, ContentType.create("application/json")))
      }
      b.clear()
      (resp \ "id").extract[Long]
    } else {
      val resp = retryTransientErrors { post("/api/v1alpha/batches/create", json = batchJson) }
      val batchID = (resp \ "id").extract[Long]

      val b = new ByteArrayBuilder()

      var i = 0
      while (i < bunches.length) {
        addBunchBytes(b, bunches(i))
        val data = b.result()
        retryTransientErrors {
          post(
            s"/api/v1alpha/batches/$batchID/jobs/create",
            new ByteArrayEntity(
              data,
              ContentType.create("application/json")))
        }
        b.clear()
        i += 1
      }

      retryTransientErrors { patch(s"/api/v1alpha/batches/$batchID/close") }
      batchID
    }
    log.info(s"run: created batch $batchID")
    batchID
  }

  def run(batchJson: JObject, jobs: IndexedSeq[JObject]): JValue = {
    val batchID = create(batchJson, jobs)
    waitForBatch(batchID, false)
  }

  def waitForBatch(batchID: Long, excludeDriverJobInBatch: Boolean): JValue = {
    implicit val formats: Formats = DefaultFormats

    val start = System.nanoTime()

    while (true) {
      val batch = retryTransientErrors { get(s"/api/v1alpha/batches/$batchID") }
      val n_completed = (batch \ "n_completed").extract[Int]
      val n_jobs = (batch \ "n_jobs").extract[Int]
      if ((excludeDriverJobInBatch && n_completed == n_jobs - 1) || n_completed == n_jobs)
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

  private def createBunches(jobs: IndexedSeq[JObject]): BoxedArrayBuilder[Array[Array[Byte]]] = {
    val bunches = new BoxedArrayBuilder[Array[Array[Byte]]]()
    val bunchb = new BoxedArrayBuilder[Array[Byte]]()

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
    bunches
  }

  private def addBunchBytes(b: ByteArrayBuilder, bunch: Array[Array[Byte]]) {
    var j = 0
    b += '['
    while (j < bunch.length) {
      if (j > 0)
        b += ','
      b ++= bunch(j)
      j += 1
    }
    b += ']'
  }
}
