package is.hail.services

import is.hail.shadedazure.com.azure.core.credential.TokenRequestContext
import is.hail.shadedazure.com.azure.identity.{
  DefaultAzureCredentialBuilder, DefaultAzureCredential
}
import is.hail.utils._

import java.io.InputStream
import scala.collection.JavaConverters._

import com.google.auth.oauth2.GoogleCredentials
import org.apache.commons.io.IOUtils
import org.apache.http.{HttpEntity, HttpEntityEnclosingRequest}
import org.apache.http.client.config.RequestConfig
import org.apache.http.client.methods.HttpUriRequest
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.util.EntityUtils
import org.apache.log4j.{LogManager, Logger}
import org.json4s.JValue
import org.json4s.jackson.JsonMethods

abstract class CloudCredentials {
  def accessToken(): String
}

class GoogleCloudCredentials() extends CloudCredentials {
  private[this] val credentials = GoogleCredentials.getApplicationDefault()

  override def accessToken(): String = {
    credentials.refreshIfExpired()
    credentials.getAccessToken.getTokenValue
  }
}

class AzureCloudCredentials() extends CloudCredentials {
  private[this] val credentials: DefaultAzureCredential =
      new DefaultAzureCredentialBuilder().build()

  override def accessToken(): String = {
    val context = new TokenRequestContext()
    context.addScopes("https://management.azure.com/.default")
    credentials.getToken(context).block().getToken()
  }
}

class ClientResponseException(
  val status: Int,
  message: String,
  cause: Throwable,
) extends Exception(message, cause) {
  def this(statusCode: Int) = this(statusCode, null, null)

  def this(statusCode: Int, message: String) = this(statusCode, message, null)
}

object Requester {
  private val log: Logger = LogManager.getLogger("Requester")
  private[this] val TIMEOUT_MS = 5 * 1000

  val httpClient: CloseableHttpClient = {
    log.info("creating HttpClient")
    val requestConfig = RequestConfig.custom()
      .setConnectTimeout(TIMEOUT_MS)
      .setConnectionRequestTimeout(TIMEOUT_MS)
      .setSocketTimeout(TIMEOUT_MS)
      .build()
    try {
      HttpClients.custom()
        .setSSLContext(tls.getSSLContext)
        .setMaxConnPerRoute(20)
        .setMaxConnTotal(100)
        .setDefaultRequestConfig(requestConfig)
        .build()
    } catch {
      case _: NoSSLConfigFound =>
        log.info("creating HttpClient with no SSL Context")
        HttpClients.custom()
          .setMaxConnPerRoute(20)
          .setMaxConnTotal(100)
          .setDefaultRequestConfig(requestConfig)
          .build()
    }
  }

  def fromEnv() = {
    val credentials = sys.env.get("HAIL_CLOUD") match {
      case Some("gcp") => new GoogleCloudCredentials()
      case Some("azure") => new AzureCloudCredentials()
      case Some(cloud) =>
        throw new IllegalArgumentException(s"Bad cloud: $cloud")
      case None =>
        throw new IllegalArgumentException(s"HAIL_CLOUD must be set.")
    }
    new Requester(credentials)
  }
}

class Requester(
  val credentials: CloudCredentials
) {
  import Requester._

  def requestWithHandler[T >: Null](req: HttpUriRequest, body: HttpEntity, f: InputStream => T)
    : T = {
    log.info(s"request ${req.getMethod} ${req.getURI}")

    if (body != null)
      req.asInstanceOf[HttpEntityEnclosingRequest].setEntity(body)

    val token = credentials.accessToken()
    req.addHeader("Authorization", s"Bearer $token")

    retryTransientErrors {
      using(httpClient.execute(req)) { resp =>
        val statusCode = resp.getStatusLine.getStatusCode
        log.info(s"request ${req.getMethod} ${req.getURI} response $statusCode")
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
          using(entity.getContent)(f)
        } else
          null
      }
    }
  }

  def requestAsByteStream(req: HttpUriRequest, body: HttpEntity = null): Array[Byte] =
    requestWithHandler(req, body, IOUtils.toByteArray)

  def request(req: HttpUriRequest, body: HttpEntity = null): JValue =
    requestWithHandler(
      req,
      body,
      { content =>
        val s = IOUtils.toByteArray(content)
        if (s.isEmpty)
          null
        else
          JsonMethods.parse(new String(s))
      },
    )
}
