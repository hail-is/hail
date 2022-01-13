package is.hail.services

import is.hail.utils._
import java.io.{File, FileInputStream}

import org.apache.http.client.methods.HttpUriRequest
import org.apache.log4j.{LogManager, Logger}
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

object Tokens {
  lazy val log: Logger = LogManager.getLogger("Tokens")

  def get: Tokens = {
    val file = getTokensFile()
    if (new File(file).isFile) {
      using(new FileInputStream(file)) { is =>
        implicit val formats: Formats = DefaultFormats
        val tokens = JsonMethods.parse(is).extract[Map[String, String]]
        log.info(s"tokens found for namespaces {${ tokens.keys.mkString(", ") }}")
        new Tokens(tokens)
      }
    } else {
      log.info(s"tokens file not found: $file")
      new Tokens(Map())
    }
  }

  def getTokensFile(): String = {
    val file = System.getenv("HAIL_TOKENS_FILE")
    if (file != null)
      file
    else if (DeployConfig.get.location == "external")
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
    val token = namespaceToken(ns)
    req.addHeader("Authorization", s"Bearer $token")
    val location = DeployConfig.get.location
    if (location == "external" && ns != "default")
      req.addHeader("X-Hail-Internal-Authorization", s"Bearer ${ namespaceToken("default") }")
  }

  def addServiceAuthHeaders(service: String, req: HttpUriRequest): Unit = {
    addNamespaceAuthHeaders(DeployConfig.get.getServiceNamespace(service), req)
  }
}
