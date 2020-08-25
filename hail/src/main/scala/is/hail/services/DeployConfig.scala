package is.hail.services

import java.io.{File, FileInputStream}
import java.net.Socket

import is.hail.utils._
import is.hail.services.tls._
import org.json4s._
import org.json4s.jackson.JsonMethods
import org.apache.http.client.methods._
import org.apache.log4j.Logger

import scala.util.Random

object DeployConfig {
  private[this] val log = Logger.getLogger("DeployConfig")

  lazy val get: DeployConfig = fromConfigFile()

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
      new DeployConfig(
        "external",
        "default",
        Map())
  }

  def fromConfig(config: JValue): DeployConfig = {
    implicit val formats: Formats = DefaultFormats
    new DeployConfig(
      (config \ "location").extract[String],
      (config \ "default_namespace").extract[String],
      (config \ "service_namespace").extract[Map[String, String]])
  }
}

class DeployConfig(
  val location: String,
  val defaultNamespace: String,
  val serviceNamespace: Map[String, String]) {
  import DeployConfig._

  def scheme(baseScheme: String = "http"): String = {
    if (location == "external" || location == "k8s")
      baseScheme + "s"
    else
      baseScheme
  }

  def getServiceNamespace(service: String): String = {
    serviceNamespace.getOrElse(service, defaultNamespace)
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
          s"$service.hail.is"
        else
          "internal.hail.is"
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

  private[this] lazy val addressRequester = new Requester("address")

  def addresses(service: String): Seq[(String, Int)] = {
    implicit val formats: Formats = DefaultFormats

    val addressBaseUrl = baseUrl("address")
    val url = s"${addressBaseUrl}/api/${service}"
    val addresses = addressRequester.request(new HttpGet(url))
      .asInstanceOf[JArray]
      .children
      .asInstanceOf[List[JObject]]
    addresses.map(x => ((x \ "address").extract[String], (x \ "port").extract[Int]))
  }

  def address(service: String): (String, Int) = {
    val serviceAddresses = addresses(service)
    val n = serviceAddresses.length
    assert(n > 0)
    serviceAddresses(Random.nextInt(n))
  }

  def socket(service: String): Socket = {
    val (host, port) = location match {
      case "k8s" | "gce" =>
        address(service)
      case "external" =>
        throw new IllegalStateException(
          s"Cannot open a socket from an external client to a service.")
    }
    log.info(s"attempting to connect ${service} at ${host}:${port}")
    val s = getSSLContext.getSocketFactory().createSocket(host, port)
    log.info(s"connected to ${service} at ${host}:${port}")
    s
  }
}
