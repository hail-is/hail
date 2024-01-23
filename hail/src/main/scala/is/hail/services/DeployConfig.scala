package is.hail.services

import is.hail.utils._

import java.io.{File, FileInputStream}

import org.apache.log4j.Logger
import org.json4s._
import org.json4s.jackson.JsonMethods

object DeployConfig {
  private[this] val log = Logger.getLogger("DeployConfig")

  private[this] lazy val default: DeployConfig = fromConfigFile()
  private[this] var _get: DeployConfig = null

  def set(x: DeployConfig) =
    _get = x

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
      val fromHome = s"${System.getenv("HOME")}/.hail/deploy-config.json"
      if (new File(fromHome).exists())
        file = fromHome
    }

    if (file == null) {
      val f = "/deploy-config/deploy-config.json"
      if (new File(f).exists())
        file = f
    }

    if (file != null) {
      using(new FileInputStream(file))(in => fromConfig(JsonMethods.parse(in)))
    } else
      fromConfig("external", "default", "hail.is", None)
  }

  def fromConfig(config: JValue): DeployConfig = {
    implicit val formats: Formats = DefaultFormats
    fromConfig(
      (config \ "location").extract[String],
      (config \ "default_namespace").extract[String],
      (config \ "domain").extract[Option[String]].getOrElse("hail.is"),
      (config \ "base_path").extract[Option[String]],
    )
  }

  def fromConfig(
    locationFromConfig: String,
    defaultNamespaceFromConfig: String,
    domainFromConfig: String,
    basePathFromConfig: Option[String],
  ): DeployConfig = {
    val location = sys.env.getOrElse(toEnvVarName("location"), locationFromConfig)
    val defaultNamespace =
      sys.env.getOrElse(toEnvVarName("default_namespace"), defaultNamespaceFromConfig)
    val domain = sys.env.getOrElse(toEnvVarName("domain"), domainFromConfig)
    val basePath = sys.env.get(toEnvVarName("basePath")).orElse(basePathFromConfig)

    (basePath, defaultNamespace) match {
      case (None, ns) if ns != "default" =>
        new DeployConfig(location, ns, s"internal.$domain", Some(s"/$ns"))
      case _ => new DeployConfig(location, defaultNamespace, domain, basePath)
    }
  }

  private[this] def toEnvVarName(s: String): String =
    "HAIL_" + s.toUpperCase
}

class DeployConfig(
  val location: String,
  val defaultNamespace: String,
  val domain: String,
  val basePath: Option[String],
) {

  def scheme(baseScheme: String = "http"): String =
    if (location == "external" || location == "k8s")
      baseScheme + "s"
    else
      baseScheme

  def domain(service: String): String = {
    location match {
      case "k8s" =>
        s"$service.$defaultNamespace"
      case "gce" =>
        if (basePath.isEmpty)
          s"$service.hail"
        else
          "internal.hail"
      case "external" =>
        if (basePath.isEmpty)
          s"$service.$domain"
        else
          domain
    }
  }

  def basePath(service: String): String =
    basePath match {
      case Some(base) => s"$base/$service"
      case None => ""
    }

  def baseUrl(service: String, baseScheme: String = "http"): String =
    s"${scheme(baseScheme)}://${domain(service)}${basePath(service)}"
}
