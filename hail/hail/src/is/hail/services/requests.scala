package is.hail.services

import is.hail.services.oauth2.CloudCredentials
import is.hail.utils._

import java.net.URL

import org.apache.http.{HttpEntity, HttpEntityEnclosingRequest}
import org.apache.http.client.config.RequestConfig
import org.apache.http.client.methods.{HttpGet, HttpPatch, HttpPost, HttpUriRequest}
import org.apache.http.entity.ContentType.APPLICATION_JSON
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.util.EntityUtils
import org.json4s.JValue
import org.json4s.JsonAST.JNothing
import org.json4s.jackson.JsonMethods

object requests extends Logging {

  class ClientResponseException(val status: Int, message: String) extends Exception(message)

  trait Requester extends AutoCloseable {
    def url: URL
    def get(route: String): JValue
    def post(route: String, body: JValue): JValue
    def post(route: String, body: HttpEntity): JValue
    def patch(route: String): JValue
  }

  private[this] val TimeoutMs = 5 * 1000
  private[this] val MaxNumConnectionPerRoute = 20
  private[this] val MaxNumConnections = 100

  def Requester(baseUrl: URL, cred: CloudCredentials): Requester = {

    val httpClient: CloseableHttpClient = {
      logger.info("creating HttpClient")
      val requestConfig = RequestConfig.custom()
        .setConnectTimeout(TimeoutMs)
        .setConnectionRequestTimeout(TimeoutMs)
        .setSocketTimeout(TimeoutMs)
        .build()
      try {
        HttpClients.custom()
          .setSSLContext(tls.getSSLContext)
          .setMaxConnPerRoute(MaxNumConnectionPerRoute)
          .setMaxConnTotal(MaxNumConnections)
          .setDefaultRequestConfig(requestConfig)
          .build()
      } catch {
        case _: NoSSLConfigFound =>
          logger.info("creating HttpClient with no SSL Context")
          HttpClients.custom()
            .setMaxConnPerRoute(MaxNumConnectionPerRoute)
            .setMaxConnTotal(MaxNumConnections)
            .setDefaultRequestConfig(requestConfig)
            .build()
      }
    }

    def request(req: HttpUriRequest, body: Option[HttpEntity] = None): JValue = {
      req.addHeader("Authorization", s"Bearer ${cred.accessToken}")
      body.foreach(entity => req.asInstanceOf[HttpEntityEnclosingRequest].setEntity(entity))
      retryTransientErrors {
        using(httpClient.execute(req)) { resp =>
          val statusCode = resp.getStatusLine.getStatusCode
          val message = Option(resp.getEntity).map(EntityUtils.toString).filter(_.nonEmpty)
          if (statusCode < 200 || statusCode >= 300) {
            logger.warn(s"$statusCode ${req.getMethod} ${req.getURI}\n${message.orNull}")
            throw new ClientResponseException(statusCode, message.orNull)
          }

          logger.info(s"$statusCode ${req.getMethod} ${req.getURI}")
          message.map(JsonMethods.parse(_)).getOrElse(JNothing)
        }
      }
    }

    new Requester {
      override val url: URL = baseUrl

      override def get(route: String): JValue =
        request(new HttpGet(s"$baseUrl$route"))

      override def post(route: String, body: JValue): JValue =
        post(route, new StringEntity(JsonMethods.compact(body), APPLICATION_JSON))

      override def post(route: String, body: HttpEntity): JValue =
        request(new HttpPost(s"$baseUrl$route"), Some(body))

      override def patch(route: String): JValue =
        request(new HttpPatch(s"$baseUrl$route"))

      override def close(): Unit =
        httpClient.close()
    }
  }
}
