package is.hail.services

import is.hail.utils._
import java.io.{File, FileInputStream}

import org.apache.http.client.methods.HttpUriRequest
import org.apache.log4j.{LogManager, Logger}
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

object Tokens {
  private[this] val log: Logger = LogManager.getLogger("Tokens")

  private[this] var _get: Tokens = null

  def set(x: Tokens) = {
    _get = x
  }

  def get: Tokens = {
    if (_get == null) {
      val file = getTokensFile()
      if (new File(file).isFile) {
        _get = fromFile(file)
      } else {
        log.info(s"tokens file not found: $file")
        _get = new Tokens(Map())
      }
    }
    return _get
  }

  def fromFile(file: String): Tokens = {
    using(new FileInputStream(file)) { is =>
      implicit val formats: Formats = DefaultFormats
      val tokens = JsonMethods.parse(is).extract[Map[String, String]]
      log.info(s"tokens found for namespaces {${ tokens.keys.mkString(", ") }}")
      new Tokens(tokens)
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
