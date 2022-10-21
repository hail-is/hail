package is.hail.services

import java.io.{File, FileInputStream}
import java.net._

import is.hail.utils._
import is.hail.services.tls._
import org.json4s._
import org.json4s.jackson.JsonMethods
import org.apache.http.client.methods._
import org.apache.log4j.Logger

import scala.util.Random

object DeployConfig {
  private[this] val log = Logger.getLogger("DeployConfig")

  private[this] lazy val default: DeployConfig = fromConfigFile()
  private[this] var _get: DeployConfig = null

  def set(x: DeployConfig) = {
    _get = x
  }

  def get(): DeployConfig = {
    if (_get == null) {
      _get = default
    }
    _get
  }

  def fromConfigFile(file0: String = null): DeployConfig = {
    var file = file0

    if (file == null)
      file = System.getenv("HAIL_DEPLOY_CONFIG_FILE")

    if (file == null) {
      val fromHome = s"${ System.getenv("HOME") }/.hail/deploy-config.json"
      if (new File(fromHome).exists())
        file = fromHome
    }

    if (file == null) {
      val f = "/deploy-config/deploy-config.json"
      if (new File(f).exists())
        file = f
    }

    if (file != null) {
      using(new FileInputStream(file)) { in =>
        fromConfig(JsonMethods.parse(in))
      }
    } else
      fromConfig("external", "default", "hail.is")
  }

  def fromConfig(config: JValue): DeployConfig = {
    implicit val formats: Formats = DefaultFormats

    val location = (config \ "location").extract[String]
    val defaultNamespace = (config \ "default_namespace").extract[String]
    val domain = (config \ "domain").extract[Option[String]].getOrElse("hail.is")

    sys.env.get("HAIL_TERRA") match {
      case Some(_) => new TerraDeployConfig(location, defaultNamespace, domain, (config \ "subpath").extract[String])
      case None => fromConfig(location, defaultNamespace, domain)
    }
  }

  def fromConfig(location: String, defaultNamespace: String, domain: String): DeployConfig = {
    new DeployConfig(
      sys.env.getOrElse(toEnvVarName("location"), location),
      sys.env.getOrElse(toEnvVarName("default_namespace"), defaultNamespace),
      sys.env.getOrElse(toEnvVarName("domain"), domain))
  }

  private[this] def toEnvVarName(s: String): String = {
    "HAIL_" + s.toUpperCase
  }
}

class DeployConfig(
  val location: String,
  val defaultNamespace: String,
  val domain: String) {
  import DeployConfig._

  def scheme(baseScheme: String = "http"): String = {
    if (location == "external" || location == "k8s")
      baseScheme + "s"
    else
      baseScheme
  }

  def getServiceNamespace(service: String): String = {
    defaultNamespace
  }

  def domain(service: String): String = {
    val ns = getServiceNamespace(service)
    location match {
      case "k8s" =>
        s"$service.$ns"
      case "gce" =>
        if (ns == "default")
          s"$service.hail"
        else
          "internal.hail"
      case "external" =>
        if (ns == "default")
          s"$service.$domain"
        else
          s"internal.$domain"
    }
  }

  def basePath(service: String): String = {
    val ns = getServiceNamespace(service)
    if (ns == "default")
      ""
    else
      s"/$ns/$service"
  }

  def baseUrl(service: String, baseScheme: String = "http"): String = {
    s"${ scheme(baseScheme) }://${ domain(service) }${ basePath(service) }"
  }
}

class TerraDeployConfig(
  location: String,
  defaultNamespace: String,
  domain: String,
  subpath: String) extends DeployConfig(location, defaultNamespace, domain) {
  import DeployConfig._

  override def domain(service: String): String = {
    this.domain
  }

  override def basePath(service: String): String = {
    s"$subpath/$service"
  }

  override def scheme(baseScheme: String = "http"): String = {
    "https"
  }
}
