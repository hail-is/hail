package is.hail.services

import is.hail.services.oauth2.CloudCredentials
import is.hail.utils.{log, _}

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

object requests {

  class ClientResponseException(val status: Int, message: String) extends Exception(message)

  trait Requester extends AutoCloseable {
    def get(route: String): JValue
    def post(route: String, body: JValue): JValue
    def post(route: String, body: HttpEntity): JValue
    def patch(route: String): JValue
  }

  private[this] val TIMEOUT_MS = 5 * 1000

  def Requester(baseUrl: URL, cred: CloudCredentials): Requester = {

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

    def request(req: HttpUriRequest, body: Option[HttpEntity] = None): JValue = {
      log.info(s"request ${req.getMethod} ${req.getURI}")
      req.addHeader("Authorization", s"Bearer ${cred.accessToken}")
      body.foreach(entity => req.asInstanceOf[HttpEntityEnclosingRequest].setEntity(entity))
      retryTransientErrors {
        using(httpClient.execute(req)) { resp =>
          val statusCode = resp.getStatusLine.getStatusCode
          log.info(s"request ${req.getMethod} ${req.getURI} response $statusCode")
          val message = Option(resp.getEntity).map(EntityUtils.toString).filter(_.nonEmpty)
          if (statusCode < 200 || statusCode >= 300) {
            throw new ClientResponseException(statusCode, message.orNull)
          }
          message.map(JsonMethods.parse(_)).getOrElse(JNothing)
        }
      }
    }

    new Requester with Logging {
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
