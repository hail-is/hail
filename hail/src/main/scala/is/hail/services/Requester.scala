package is.hail.services

import java.nio.charset.StandardCharsets

import is.hail.utils._
import is.hail.services._
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

class ClientResponseException(
  val status: Int,
  message: String,
  cause: Throwable
) extends Exception(message, cause) {
  def this(statusCode: Int) = this(statusCode, null, null)

  def this(statusCode: Int, message: String) = this(statusCode, message, null)
}

object Requester {
  lazy val log: Logger = LogManager.getLogger("Requester")

  private val httpClient: CloseableHttpClient = {
    log.info("creating HttpClient")
    HttpClients.custom()
      .setSSLContext(tls.getSSLContext)
      .build()
  }
}

class Requester(
  tokens: Tokens,
  val service: String
) {
  def this(service: String) = this(Tokens.get, service)

  import Requester._

  def request(req: HttpUriRequest, body: HttpEntity = null): JValue = {
    log.info(s"request ${ req.getMethod } ${ req.getURI }")

    if (body != null)
      req.asInstanceOf[HttpEntityEnclosingRequest].setEntity(body)

    tokens.addServiceAuthHeaders(service, req)

    retryTransientErrors {
      using(httpClient.execute(req)) { resp =>
        val statusCode = resp.getStatusLine.getStatusCode
        log.info(s"request ${ req.getMethod } ${ req.getURI } response $statusCode")
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
  }
}
