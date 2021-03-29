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

object Requester {
  lazy val log: Logger = LogManager.getLogger("Requester")
}

class Requester(
  tokens: Tokens,
  val service: String
) {
  def this(service: String) = this(Tokens.get, service)

  import Requester._

  def request(req: HttpUriRequest, body: HttpEntity = null): JValue = {
    if (body != null)
      req.asInstanceOf[HttpEntityEnclosingRequest].setEntity(body)

    tokens.addServiceAuthHeaders(service, req)

    val maybeString = Http.internalRequestString(req)
    if (maybeString != null) {
      JsonMethods.parse(maybeString)
    } else {
      null
    }
  }
}
