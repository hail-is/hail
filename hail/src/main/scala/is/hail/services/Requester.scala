package is.hail.services

import java.io.InputStream
import java.nio.charset.StandardCharsets

import is.hail.HailContext
import is.hail.utils._
import is.hail.services._
import is.hail.shadedazure.com.azure.identity.{ClientSecretCredential, ClientSecretCredentialBuilder}
import is.hail.shadedazure.com.azure.core.credential.TokenRequestContext

import com.google.auth.oauth2.ServiceAccountCredentials
import org.apache.commons.io.IOUtils
import org.apache.http.{HttpEntity, HttpEntityEnclosingRequest}
import org.apache.http.client.methods.{HttpDelete, HttpGet, HttpPatch, HttpPost, HttpUriRequest}
import org.apache.http.entity.{ByteArrayEntity, ContentType, StringEntity}
import org.apache.http.client.config.RequestConfig
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager
import org.apache.http.util.EntityUtils
import org.apache.log4j.{LogManager, Logger}
import org.json4s.{DefaultFormats, Formats, JObject, JValue}
import org.json4s.jackson.JsonMethods

import scala.util.Random
import java.io.FileInputStream


abstract class CloudCredentials {
  def accessToken(): String
}

class GoogleCloudCredentials(gsaKeyPath: String) extends CloudCredentials {
  private[this] val scopes = Array(
  )

  private[this] val credentials = using(new FileInputStream(gsaKeyPath)) { is =>
    ServiceAccountCredentials
      .fromStream(is)
      .createScoped(
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
      )
  }

  override def accessToken(): String = {
    credentials.refreshIfExpired()
    credentials.getAccessToken.getTokenValue
  }
}

class AzureCloudCredentials(credentialsPath: String) extends CloudCredentials {
  private[this] val credentials = using(new FileInputStream(credentialsPath)) { is =>
      implicit val formats: Formats = defaultJSONFormats
      val kvs = JsonMethods.parse(is)
      val appId = (kvs \ "appId").extract[String]
      val password = (kvs \ "password").extract[String]
      val tenant = (kvs \ "tenant").extract[String]

      new ClientSecretCredentialBuilder()
        .clientId(appId)
        .clientSecret(password)
        .tenantId(tenant)
        .build()
  }

  override def accessToken(): String = {
    val context = new TokenRequestContext()
    // TODO I hope I dont have to give this scope
    context.addScopes("https://management.azure.com/.default")
    credentials.getToken(context).block.getToken
  }
}

class ClientResponseException(
  val status: Int,
  message: String,
  cause: Throwable
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
    } catch { case _: NoSSLConfigFound =>
      log.info("creating HttpClient with no SSL Context")
      HttpClients.custom()
        .setMaxConnPerRoute(20)
        .setMaxConnTotal(100)
        .setDefaultRequestConfig(requestConfig)
        .build()
    }
  }

  def fromCredentialsFile(credentialsPath: String) = {
    val credentials = sys.env.get("HAIL_CLOUD") match {
      case Some("gcp") => new GoogleCloudCredentials(credentialsPath)
      case Some("azure") => new AzureCloudCredentials(credentialsPath)
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
  def requestWithHandler[T >: Null](req: HttpUriRequest, body: HttpEntity, f: InputStream => T): T = {
    log.info(s"request ${ req.getMethod } ${ req.getURI }")

    if (body != null)
      req.asInstanceOf[HttpEntityEnclosingRequest].setEntity(body)

    val token = credentials.accessToken()
    req.addHeader("Authorization", s"Bearer $token")

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
          using(entity.getContent)(f)
        } else
          null
      }
    }
  }

  def requestAsByteStream(req: HttpUriRequest, body: HttpEntity = null): Array[Byte] =
    requestWithHandler(req, body, IOUtils.toByteArray)

  def request(req: HttpUriRequest, body: HttpEntity = null): JValue =
    requestWithHandler(req, body, { content =>
      val s = IOUtils.toByteArray(content)
      if (s.isEmpty)
        null
      else
        JsonMethods.parse(new String(s))
    })
}
