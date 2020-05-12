package is.hail.services

import is.hail.utils._
import java.io.FileInputStream

import org.apache.http.client.methods.HttpUriRequest
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

object Tokens {
  def get: Tokens = {
    using(new FileInputStream(getTokensFile())) { is =>
      implicit val formats: Formats = DefaultFormats
      new Tokens(JsonMethods.parse(is).extract[Map[String, String]])
    }
  }

  def getTokensFile(): String = {
    if (DeployConfig.get.location == "external")
      s"${ System.getenv("HOME") }/.hail/tokens.json"
    else
      "/user-tokens/tokens.json"
  }
}

class Tokens(
  tokens: Map[String, String]
) {
  def namespaceToken(ns: String): String = tokens(ns)

  def addNamespaceAuthHeaders(ns: String, req: HttpUriRequest): Unit = {
    req.addHeader("Authorization", s"Bearer ${ namespaceToken(ns) }")
    val location = DeployConfig.get.location
    if (location == "external" && ns != "default")
      req.addHeader("X-Hail-Internal-Authorization", s"Bearer ${ namespaceToken("default") }")
  }

  def addServiceAuthHeaders(service: String, req: HttpUriRequest): Unit = {
    addNamespaceAuthHeaders(DeployConfig.get.getServiceNamespace(service), req)
  }
}
