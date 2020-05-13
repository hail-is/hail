package is.hail.services.batch_client

import java.util.Random

import is.hail.utils._
import is.hail.services.{DeployConfig, Tokens}
import org.apache.commons.io.IOUtils
import org.apache.http.{HttpEntity, HttpEntityEnclosingRequest}
import org.apache.http.client.methods.{HttpDelete, HttpGet, HttpPatch, HttpPost, HttpUriRequest}
import org.apache.http.entity.{ByteArrayEntity, ContentType, StringEntity}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.util.EntityUtils
import org.json4s.JsonAST.JNull
import org.json4s.{DefaultFormats, Formats, JObject, JValue}
import org.json4s.jackson.JsonMethods

class NoBodyException(message: String, cause: Throwable) extends Exception(message, cause) {
  def this() = this(null, null)

  def this(message: String) = this(message, null)
}

class ClientResponseException(statusCode: Int, message: String, cause: Throwable) extends Exception(message, cause) {
  def this(statusCode: Int) = this(statusCode, null, null)

  def this(statusCode: Int, message: String) = this(statusCode, message, null)
}

class BatchClient extends AutoCloseable {
  private[this] val baseUrl = DeployConfig.get.baseUrl("batch")

  private[this] val httpClient: CloseableHttpClient = HttpClients.createDefault();

  private[this] def request(req: HttpUriRequest, body: HttpEntity = null): JValue = {
    if (body != null)
      req.asInstanceOf[HttpEntityEnclosingRequest].setEntity(body)
    Tokens.get.addServiceAuthHeaders("batch", req)
    using(httpClient.execute(req)) { resp =>
      val statusCode = resp.getStatusLine.getStatusCode
      if (statusCode < 200 || statusCode >= 300) {
        val entity = resp.getEntity
        val message =
          if (entity != null)
            EntityUtils.toString(entity)
        else
            null
        throw new ClientResponseException(statusCode, message)
      }
      val entity: HttpEntity = resp.getEntity
      if (entity != null) {
        using(entity.getContent) { content =>
          val s = IOUtils.toByteArray(content)
          if (s.isEmpty)
            null
          else
            JsonMethods.parse(new String(s))
        }
      } else
        null
    }
  }

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

  def delete(path: String): JValue =
    request(new HttpDelete(s"$baseUrl$path"))

  def createJobs(batchID: Long, jobs: IndexedSeq[JObject]): Unit = {
    val bunches = new ArrayBuilder[Array[Array[Byte]]]()

    val bunchb = new ArrayBuilder[Array[Byte]]()

    var i = 0
    var size = 0
    while (i < jobs.length) {
      val jobBytes = JsonMethods.compact(jobs(i)).getBytes
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

    val random = new Random()
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
          (0.1 * (0.8 + random.nextFloat() * 0.4) * (elapsed / 1000.0 / 1000)).toInt,
          5000),
        50)
      Thread.sleep(d)
    }

    JNull // placate the type checker, cannot be reached
  }

  def close() {
    httpClient.close()
  }
}
